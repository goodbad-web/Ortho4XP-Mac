"""Shared-memory handoff primitives for the optional tile conversion path.

The normal Ortho4XP path remains file based.  This module only owns the
short-lived transport buffers used by the opt-in resident ASHelper route:
Python creates and leases the buffers, ASHelper maps them by POSIX name, and
Python validates the returned DDS bytes before publishing them to the tile
transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from multiprocessing.shared_memory import SharedMemory
import threading
import time
from typing import Dict, Optional


GIB = 1024 ** 3
MIN_AUTO_BUDGET_BYTES = 2 * GIB
MAX_AUTO_BUDGET_BYTES = 8 * GIB
DEFAULT_INPUT_PIXEL_FORMAT = "RGB8"
DEFAULT_OUTPUT_PIXEL_FORMAT = "DDS"


class SharedMemoryError(RuntimeError):
    """A shared-memory buffer cannot be created, mapped, or read safely."""


def shared_memory_budget_bytes(
    configured_gb: int = 0,
    physical_bytes: Optional[int] = None,
) -> int:
    """Return the bounded shared-memory budget for one tile.

    A configured positive value is respected.  Zero means eight percent of
    physical memory, clamped to the conservative 2--8 GiB range agreed for
    the initial implementation.  The budget is a scheduler limit, not a
    promise that all of the host's unified memory is free.
    """

    configured_gb = int(configured_gb or 0)
    if configured_gb > 0:
        return configured_gb * GIB
    if physical_bytes is None:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_bytes = pages * page_size
        except (AttributeError, OSError, ValueError):
            physical_bytes = 0
    estimated = int(max(0, int(physical_bytes or 0)) * 0.08)
    return min(MAX_AUTO_BUDGET_BYTES, max(MIN_AUTO_BUDGET_BYTES, estimated))


class SharedMemoryBudget:
    """Condition-backed byte budget used to apply backpressure before submit."""

    def __init__(self, capacity_bytes: int) -> None:
        capacity_bytes = int(capacity_bytes)
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self.in_use_bytes = 0
        self._condition = threading.Condition()

    def acquire(self, size_bytes: int, timeout: Optional[float] = None) -> bool:
        size_bytes = int(size_bytes)
        if size_bytes <= 0 or size_bytes > self.capacity_bytes:
            return False
        with self._condition:
            if timeout is None:
                while self.in_use_bytes + size_bytes > self.capacity_bytes:
                    self._condition.wait()
            else:
                remaining = max(0.0, float(timeout))
                while self.in_use_bytes + size_bytes > self.capacity_bytes:
                    if remaining <= 0:
                        return False
                    started = time.monotonic()
                    self._condition.wait(remaining)
                    remaining -= time.monotonic() - started
                    if remaining <= 0 and self.in_use_bytes + size_bytes > self.capacity_bytes:
                        return False
            self.in_use_bytes += size_bytes
            return True

    def release(self, size_bytes: int) -> None:
        size_bytes = int(size_bytes)
        with self._condition:
            self.in_use_bytes = max(0, self.in_use_bytes - max(0, size_bytes))
            self._condition.notify_all()

    def snapshot(self) -> Dict[str, int]:
        with self._condition:
            return {
                "capacity_bytes": self.capacity_bytes,
                "in_use_bytes": self.in_use_bytes,
                "available_bytes": max(0, self.capacity_bytes - self.in_use_bytes),
            }


@dataclass(frozen=True)
class SharedMemoryDescriptor:
    """JSON-compatible description of one mapped raster or DDS buffer."""

    name: str
    offset: int
    capacity: int
    used_bytes: int
    width: int
    height: int
    stride: int
    pixel_format: str
    read_only: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "capacity": self.capacity,
            "used_bytes": self.used_bytes,
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "pixel_format": self.pixel_format,
            "read_only": self.read_only,
        }


class SharedMemoryRegion:
    """Own one Python-created shared-memory segment until request completion."""

    def __init__(self, size_bytes: int, *, label: str = "buffer") -> None:
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        try:
            self._shared_memory = SharedMemory(create=True, size=size_bytes)
        except (OSError, ValueError) as error:
            raise SharedMemoryError("could not create shared memory: {}".format(error)) from error
        self.label = str(label)
        self.size_bytes = size_bytes
        self._closed = False

    @property
    def name(self) -> str:
        return self._shared_memory.name

    @property
    def buffer(self):
        if self._closed:
            raise SharedMemoryError("shared memory region is closed")
        return self._shared_memory.buf

    def write(self, payload: bytes) -> None:
        if len(payload) > self.size_bytes:
            raise SharedMemoryError(
                "{} payload is {} bytes, capacity is {}".format(
                    self.label, len(payload), self.size_bytes
                )
            )
        self.buffer[: len(payload)] = payload

    def read(self, used_bytes: int) -> bytes:
        used_bytes = int(used_bytes)
        if used_bytes < 0 or used_bytes > self.size_bytes:
            raise SharedMemoryError(
                "{} returned invalid byte count {}".format(self.label, used_bytes)
            )
        return bytes(self.buffer[:used_bytes])

    def descriptor(
        self,
        *,
        width: int = 0,
        height: int = 0,
        stride: int = 0,
        pixel_format: str,
        used_bytes: int,
        read_only: bool,
    ) -> Dict[str, object]:
        return SharedMemoryDescriptor(
            name=self.name,
            offset=0,
            capacity=self.size_bytes,
            used_bytes=int(used_bytes),
            width=int(width),
            height=int(height),
            stride=int(stride),
            pixel_format=str(pixel_format),
            read_only=bool(read_only),
        ).as_dict()

    def close(self, *, unlink: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shared_memory.close()
        finally:
            if unlink:
                try:
                    self._shared_memory.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    # Cleanup is best effort after the mapping is closed.  The
                    # caller still gets the original conversion result.
                    pass

    def __enter__(self) -> "SharedMemoryRegion":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def dds_capacity_bytes(width: int, height: int, pixel_format: str) -> int:
    """Return an exact DDS capacity for BC1/BC3 including the full mip chain."""

    width = int(width)
    height = int(height)
    normalized = str(pixel_format).upper()
    if width <= 0 or height <= 0 or normalized not in ("BC1", "BC3"):
        raise ValueError("invalid DDS geometry or format")
    block_bytes = 8 if normalized == "BC1" else 16
    payload = 0
    level_width, level_height = width, height
    while True:
        payload += max(1, (level_width + 3) // 4) * max(1, (level_height + 3) // 4) * block_bytes
        if level_width == 1 and level_height == 1:
            break
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    return 128 + payload
