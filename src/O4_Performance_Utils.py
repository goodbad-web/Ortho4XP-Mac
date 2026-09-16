"""Low-overhead performance and execution metrics for tile builds.

The module deliberately has no dependency on the GUI, imagery, or tile
modules.  It can therefore be used from the main process and from small test
fixtures without importing the full application graph.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import resource
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, Optional


SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    """Return the current process peak resident set size in MiB.

    macOS reports ``ru_maxrss`` in bytes while Linux reports KiB.  The
    conversion is kept local so the metrics schema remains platform-neutral.
    """

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON document atomically beside its final path."""

    destination = os.path.abspath(path)
    parent = os.path.dirname(destination) or os.curdir
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(destination) + ".",
        suffix=".tmp",
        dir=parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


class PerformanceMetrics:
    """Thread-safe metrics collector for one All in one execution."""

    def __init__(self, tile: Any = None, mode: str = "all_in_one") -> None:
        self._lock = threading.RLock()
        self._started_monotonic = time.perf_counter()
        self._stage_starts: Dict[str, float] = {}
        self._active_attempt: Optional[Dict[str, Any]] = None
        self.data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tile": {
                "lat": getattr(tile, "lat", None),
                "lon": getattr(tile, "lon", None),
            },
            "mode": mode,
            "started_at": _timestamp(),
            "config": {},
            "capabilities": {},
            "attempts": [],
            "totals": {
                "duration_ms": 0.0,
                "peak_rss_mb": peak_rss_mb(),
                "counters": {},
                "batches": {},
                "queue": {},
            },
            "failure": None,
        }

    def set_config(self, values: Dict[str, Any]) -> None:
        with self._lock:
            self.data["config"].update(deepcopy(values))

    def set_capabilities(self, values: Dict[str, Any]) -> None:
        with self._lock:
            self.data["capabilities"].update(deepcopy(values))

    def begin_attempt(
        self,
        index: int,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if self._active_attempt is not None:
                self.end_attempt("interrupted")
            self._active_attempt = {
                "index": int(index),
                "started_at": _timestamp(),
                "settings": deepcopy(settings or {}),
                "stages": {},
                "counters": {},
                "batches": {},
                "queue": {},
                "result": None,
            }

    def end_attempt(self, result: str) -> None:
        with self._lock:
            if self._active_attempt is None:
                return
            self._active_attempt["result"] = str(result)
            self._active_attempt["ended_at"] = _timestamp()
            self.data["attempts"].append(self._active_attempt)
            self._active_attempt = None

    def _target(self) -> Dict[str, Any]:
        return self._active_attempt or self.data["totals"]

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.record_stage(name, elapsed_ms)

    def record_stage(self, name: str, duration_ms: float) -> None:
        with self._lock:
            target = self._target()
            targets = [target]
            if target is not self.data["totals"]:
                targets.append(self.data["totals"])
            for current in targets:
                stage = current.setdefault("stages", {}).setdefault(
                    str(name), {"duration_ms": 0.0, "runs": 0}
                )
                stage["duration_ms"] += float(duration_ms)
                stage["runs"] += 1
            self.data["totals"]["peak_rss_mb"] = max(
                float(self.data["totals"].get("peak_rss_mb", 0.0)), peak_rss_mb()
            )

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            target = self._target()
            targets = [target]
            if target is not self.data["totals"]:
                targets.append(self.data["totals"])
            for current in targets:
                counters = current.setdefault("counters", {})
                counters[str(name)] = counters.get(str(name), 0) + int(amount)

    def record_batch(
        self,
        name: str,
        count: int,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        with self._lock:
            target = self._target()
            targets = [target]
            if target is not self.data["totals"]:
                targets.append(self.data["totals"])
            for current in targets:
                batch = current.setdefault("batches", {}).setdefault(
                    str(name),
                    {"batches": 0, "items": 0, "duration_ms": 0.0, "statuses": {}},
                )
                batch["batches"] += 1
                batch["items"] += int(count)
                if duration_ms is not None:
                    batch["duration_ms"] += float(duration_ms)
                if status:
                    statuses = batch.setdefault("statuses", {})
                    statuses[status] = statuses.get(status, 0) + 1

    def record_queue(self, name: str, size: int, capacity: Optional[int] = None) -> None:
        with self._lock:
            target = self._target()
            targets = [target]
            if target is not self.data["totals"]:
                targets.append(self.data["totals"])
            for current in targets:
                queue_data = current.setdefault("queue", {}).setdefault(
                    str(name), {"samples": 0, "max_size": 0}
                )
                queue_data["samples"] += 1
                queue_data["max_size"] = max(queue_data["max_size"], int(size))
                queue_data["last_size"] = int(size)
                if capacity is not None:
                    queue_data["capacity"] = int(capacity)

    def fail(self, error: Any) -> None:
        with self._lock:
            self.data["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = deepcopy(self.data)
            if self._active_attempt is not None:
                snapshot["active_attempt"] = deepcopy(self._active_attempt)
            return snapshot

    def write(self, path: str, finished: bool = True) -> None:
        with self._lock:
            if finished:
                self.data["ended_at"] = _timestamp()
                self.data["totals"]["duration_ms"] = (
                    time.perf_counter() - self._started_monotonic
                ) * 1000.0
                self.data["totals"]["peak_rss_mb"] = max(
                    float(self.data["totals"].get("peak_rss_mb", 0.0)),
                    peak_rss_mb(),
                )
            payload = self.snapshot()
        atomic_write_json(path, payload)
