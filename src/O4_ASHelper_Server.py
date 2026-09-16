"""Resident ASHelper JSON Lines client.

The client owns one helper process for one tile conversion runner.  Requests
are synchronous by design: the tile scheduler has one GPU route, so a single
request can be matched to a single response without adding another queue or
reordering task results.  CPU conversion remains the final fallback.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


_STDOUT_EOF = object()


class ASHelperServerError(RuntimeError):
    """Base error for a resident ASHelper protocol failure."""


class ASHelperServerCrashed(ASHelperServerError):
    """The helper exited or returned an invalid JSON Lines response."""

    def __init__(self, message: str, *, restarted: bool = False) -> None:
        super().__init__(message)
        self.restarted = bool(restarted)


class ASHelperJSONLServer:
    """Keep one ``ASHelper --serve`` process alive for a tile.

    A process crash permits exactly one restart.  The request that was in
    flight is never replayed; the caller can send that task batch to the CPU
    fallback.  If the restarted process also crashes, GPU work is disabled for
    the rest of the tile and subsequent requests fail immediately.
    """

    def __init__(
        self,
        executable: str,
        *,
        max_restarts: int = 1,
        logger: Optional[Callable[[str], None]] = None,
        environment: Optional[Mapping[str, str]] = None,
        response_timeout_s: float = 300.0,
    ) -> None:
        if float(response_timeout_s) <= 0:
            raise ValueError("response_timeout_s must be positive")
        self.executable = os.path.abspath(executable)
        self.max_restarts = max(0, int(max_restarts))
        self.logger = logger or (lambda _message: None)
        self.environment = dict(environment) if environment is not None else None
        self.response_timeout_s = float(response_timeout_s)
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_queue = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._restart_count = 0
        self._gpu_disabled = False
        self._closed = False

    @property
    def gpu_disabled(self) -> bool:
        return self._gpu_disabled

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise ASHelperServerError("ASHelper server is closed")
            if self._gpu_disabled:
                raise ASHelperServerError("ASHelper GPU route is disabled for this tile")
            if self._process is not None and self._process.poll() is None:
                return
            if not os.path.isfile(self.executable):
                raise ASHelperServerError(
                    "ASHelper executable is missing: {}".format(self.executable)
                )
            if not os.access(self.executable, os.X_OK):
                raise ASHelperServerError(
                    "ASHelper executable is not executable: {}".format(self.executable)
                )
            process_environment = os.environ.copy()
            if self.environment is not None:
                process_environment.update(self.environment)
            self._process = subprocess.Popen(
                [self.executable, "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=process_environment,
            )
            stdout_queue = queue.Queue()
            self._stdout_queue = stdout_queue
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(self._process, stdout_queue),
                name="Ortho4XP-ASHelper-stdout",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(self._process,),
                name="Ortho4XP-ASHelper-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

    def _read_stdout(self, process: subprocess.Popen[str], stdout_queue) -> None:
        stream = process.stdout
        if stream is None:
            stdout_queue.put(_STDOUT_EOF)
            return
        try:
            for line in stream:
                stdout_queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            stdout_queue.put(_STDOUT_EOF)

    def _drain_stderr(self, process: subprocess.Popen[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                message = line.rstrip()
                if message:
                    self.logger("ASHelper: " + message)
        except (OSError, ValueError):
            return

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._stdout_queue = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        except (OSError, ValueError):
            pass

    def _handle_crash(self, reason: str) -> ASHelperServerCrashed:
        self._stop_process()
        if self._restart_count < self.max_restarts:
            self._restart_count += 1
            try:
                self.start()
            except Exception as restart_error:
                self._gpu_disabled = True
                return ASHelperServerCrashed(
                    "{}; ASHelper restart failed: {}".format(reason, restart_error),
                    restarted=False,
                )
            self.logger(
                "WARNING: ASHelper server restarted after a protocol/process failure "
                "({}/{})".format(self._restart_count, self.max_restarts)
            )
            return ASHelperServerCrashed(reason, restarted=True)
        self._gpu_disabled = True
        self.logger(
            "WARNING: ASHelper server failed again; disabling GPU conversion for this tile."
        )
        return ASHelperServerCrashed(reason, restarted=False)

    def request(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Send one request and return its decoded response.

        The caller decides whether per-task failures should be retried by CPU.
        A process/protocol failure raises ``ASHelperServerCrashed`` and never
        replays the in-flight request.
        """

        with self._lock:
            self.start()
            process = self._process
            response_queue = self._stdout_queue
            if process is None or process.stdin is None or response_queue is None:
                raise ASHelperServerError("ASHelper server streams are unavailable")
            request_id = payload.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("ASHelper request id must be a non-empty string")
            try:
                process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = response_queue.get(timeout=self.response_timeout_s)
                if line is _STDOUT_EOF:
                    raise ASHelperServerCrashed(
                        "ASHelper server exited while processing request"
                    )
                response = json.loads(line)
                if not isinstance(response, dict):
                    raise ASHelperServerCrashed("ASHelper server returned a non-object response")
                if response.get("id") != request_id:
                    raise ASHelperServerCrashed(
                        "ASHelper response id mismatch: expected {}, got {}".format(
                            request_id, response.get("id")
                        )
                    )
                return response
            except queue.Empty as error:
                raise self._handle_crash(
                    "ASHelper server response timed out after {:.1f}s".format(
                        self.response_timeout_s
                    )
                ) from error
            except ASHelperServerCrashed as error:
                if error.restarted:
                    raise
                raise self._handle_crash(str(error)) from error
            except (BrokenPipeError, OSError, ValueError, json.JSONDecodeError) as error:
                raise self._handle_crash(
                    "ASHelper server protocol failure: {}".format(type(error).__name__)
                ) from error

    def convert_batch(
        self,
        tasks: Iterable[Mapping[str, Any]],
        *,
        gpu: bool = True,
    ) -> Dict[str, Any]:
        task_list = [dict(task) for task in tasks]
        return self.request(
            {
                "id": "batch-{}".format(uuid.uuid4().hex),
                "op": "convert_batch",
                "gpu": bool(gpu),
                "tasks": task_list,
            }
        )

    def metalfx_upscale_batch(
        self,
        tasks: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return self.request(
            {
                "id": "batch-{}".format(uuid.uuid4().hex),
                "op": "metalfx_upscale_batch",
                "gpu": True,
                "tasks": [dict(task) for task in tasks],
            }
        )

    def tensorops_upscale_batch(
        self,
        pack: str,
        tasks: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return self.request(
            {
                "id": "batch-{}".format(uuid.uuid4().hex),
                "op": "tensorops_upscale_batch",
                "gpu": True,
                "pack": pack,
                "tasks": [dict(task) for task in tasks],
            }
        )

    def mask_blur_batch(
        self,
        tasks: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return raster_batch(self, "mask_blur_batch", tasks)

    def dem_smooth_batch(
        self,
        tasks: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return raster_batch(self, "dem_smooth_batch", tasks)

    def disable_gpu(self) -> None:
        with self._lock:
            self._gpu_disabled = True
            self._stop_process()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            if process is not None and process.poll() is None and not self._gpu_disabled:
                try:
                    self.request(
                        {
                            "id": "shutdown-{}".format(uuid.uuid4().hex),
                            "op": "shutdown",
                        }
                    )
                except (ASHelperServerError, OSError, ValueError):
                    pass
            self._closed = True
            self._stop_process()

    def __enter__(self) -> "ASHelperJSONLServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def raster_batch(
    server: ASHelperJSONLServer,
    operation: str,
    tasks: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Send a raw-raster batch while keeping the protocol details local."""
    return server.request(
        {
            "id": "batch-{}".format(uuid.uuid4().hex),
            "op": operation,
            "gpu": True,
            "tasks": [dict(task) for task in tasks],
        }
    )
