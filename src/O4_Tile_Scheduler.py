"""Bounded conversion scheduling for one Ortho4XP tile.

The scheduler intentionally does not import the imagery or GUI modules.  A
caller supplies CPU/GPU dispatch functions, which keeps the queue semantics
deterministic in unit tests and prevents a second copy of conversion policy
from growing here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


_CLOSE = object()


@dataclass(frozen=True)
class ConversionTask:
    """One conversion request produced after an imagery download."""

    task_id: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversionResult:
    task_id: str
    ok: bool
    backend: str
    error: Optional[str] = None
    fallback: bool = False


Dispatch = Callable[[Sequence[ConversionTask]], Any]
Eligible = Callable[[ConversionTask], bool]
Logger = Callable[[str], None]


class TileConversionScheduler:
    """Route downloaded texture tasks to bounded CPU/GPU workers.

    ``dispatch_cpu`` and ``dispatch_gpu`` may return any of the following:

    * ``True``: every task succeeded;
    * ``False``: every task failed;
    * a mapping from task id to bool/``ConversionResult``;
    * a sequence of bool/``ConversionResult`` values matching the input order.

    GPU failures are retried only for failed tasks through the CPU dispatcher.
    A caller can therefore preserve prepared inputs and avoid a second upscale.
    """

    def __init__(
        self,
        dispatch_cpu: Dispatch,
        dispatch_gpu: Optional[Dispatch] = None,
        gpu_eligible: Optional[Eligible] = None,
        *,
        queue_size: int = 32,
        cpu_batch_size: int = 1,
        gpu_batch_size: int = 32,
        batch_wait_ms: int = 50,
        metrics: Any = None,
        logger: Optional[Logger] = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if cpu_batch_size < 1:
            raise ValueError("cpu_batch_size must be positive")
        if gpu_batch_size < 1:
            raise ValueError("gpu_batch_size must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must not be negative")
        self.dispatch_cpu = dispatch_cpu
        self.dispatch_gpu = dispatch_gpu
        self.gpu_eligible = gpu_eligible or (lambda _task: False)
        self.queue_size = int(queue_size)
        self.cpu_batch_size = int(cpu_batch_size)
        self.gpu_batch_size = int(gpu_batch_size)
        self.batch_wait_seconds = float(batch_wait_ms) / 1000.0
        self.metrics = metrics
        self.logger = logger or (lambda _message: None)

        self._input: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
        self._cpu: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
        self._gpu: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
        self._cancelled = threading.Event()
        self._closed = threading.Event()
        self._close_enqueued = threading.Event()
        self._started = False
        self._threads: List[threading.Thread] = []
        self._results: Dict[str, ConversionResult] = {}
        self._results_lock = threading.RLock()
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._threads = [
            threading.Thread(
                target=self._classify_loop,
                name="Ortho4XP-conversion-classifier",
                daemon=True,
            ),
            threading.Thread(
                target=self._route_loop,
                args=(self._cpu, "cpu", self.cpu_batch_size),
                name="Ortho4XP-conversion-cpu",
                daemon=True,
            ),
        ]
        if self.dispatch_gpu is not None:
            self._threads.append(
                threading.Thread(
                    target=self._route_loop,
                    args=(self._gpu, "gpu", self.gpu_batch_size),
                    name="Ortho4XP-conversion-gpu",
                    daemon=True,
                )
            )
        for thread in self._threads:
            thread.start()

    def submit(self, task: ConversionTask, timeout: Optional[float] = None) -> bool:
        """Submit a task, applying backpressure when the input is full."""

        if not self._started:
            self.start()
        if self._closed.is_set() or self._cancelled.is_set():
            self._record(task, False, "cancelled", "scheduler_closed")
            return False
        if timeout is None:
            while True:
                if self._closed.is_set() or self._cancelled.is_set():
                    self._record(task, False, "cancelled", "scheduler_closed")
                    return False
                try:
                    self._input.put(task, timeout=0.25)
                    break
                except queue.Full:
                    continue
        else:
            try:
                self._input.put(task, timeout=timeout)
            except queue.Full:
                self._record(task, False, "scheduler", "queue_full")
                return False
        self._record_queue(self._input, "input")
        return True

    def close(self) -> None:
        """Close the producer side and flush all accepted tasks."""

        if not self._started:
            self.start()
        if self._closed.is_set():
            return
        self._closed.set()
        if not self._close_enqueued.is_set():
            self._input.put(_CLOSE)
            self._close_enqueued.set()

    def cancel(self) -> None:
        """Request cooperative cancellation of not-yet-dispatched work."""

        self._cancelled.set()
        self.close()

    def wait(self, timeout: Optional[float] = None) -> List[ConversionResult]:
        """Close if necessary, wait for workers, and return results."""

        if not self._started:
            self.start()
        if not self._closed.is_set():
            self.close()
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self._threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError("conversion scheduler did not drain before timeout")
        with self._results_lock:
            return list(self._results.values())

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def result_for(self, task_id: str) -> Optional[ConversionResult]:
        with self._results_lock:
            return self._results.get(task_id)

    def _classify_loop(self) -> None:
        try:
            while True:
                item = self._input.get()
                try:
                    if item is _CLOSE:
                        if self.dispatch_gpu is not None:
                            self._gpu.put(_CLOSE)
                        self._cpu.put(_CLOSE)
                        return
                    task = item
                    if self._cancelled.is_set():
                        self._record(task, False, "cancelled", "cancelled_before_dispatch")
                        continue
                    if self.dispatch_gpu is not None and self.gpu_eligible(task):
                        self._gpu.put(task)
                        self._record_queue(self._gpu, "gpu")
                    else:
                        self._cpu.put(task)
                        self._record_queue(self._cpu, "cpu")
                finally:
                    self._input.task_done()
        except BaseException as error:  # pragma: no cover - defensive worker guard
            self._error = error
            self.logger(f"conversion classifier failed: {error}")

    def _route_loop(
        self,
        work_queue: queue.Queue[Any],
        backend: str,
        batch_size: int,
    ) -> None:
        pending: List[ConversionTask] = []
        try:
            while True:
                timeout = self.batch_wait_seconds if pending else None
                try:
                    item = work_queue.get(timeout=timeout)
                except queue.Empty:
                    self._dispatch_batch(pending, backend)
                    pending = []
                    continue
                try:
                    if item is _CLOSE:
                        if pending:
                            self._dispatch_batch(pending, backend)
                        return
                    if self._cancelled.is_set():
                        self._record(item, False, "cancelled", "cancelled_before_dispatch")
                        continue
                    pending.append(item)
                    if len(pending) >= batch_size:
                        self._dispatch_batch(pending, backend)
                        pending = []
                finally:
                    work_queue.task_done()
        except BaseException as error:  # pragma: no cover - defensive worker guard
            self._error = error
            self.logger(f"conversion {backend} worker failed: {error}")
            for task in pending:
                self._record(task, False, backend, type(error).__name__)

    def _dispatch_batch(self, tasks: Sequence[ConversionTask], backend: str) -> None:
        if not tasks:
            return
        started = time.perf_counter()
        dispatcher = self.dispatch_gpu if backend == "gpu" else self.dispatch_cpu
        if dispatcher is None:
            for task in tasks:
                self._record(task, False, backend, "dispatcher_unavailable")
            return
        try:
            raw_results = dispatcher(tasks)
            results = self._normalize_results(tasks, raw_results, backend)
        except BaseException as error:
            self.logger(f"conversion {backend} batch failed: {error}")
            results = [
                ConversionResult(task.task_id, False, backend, type(error).__name__)
                for task in tasks
            ]

        failed = [result for result in results if not result.ok]
        if backend == "gpu" and failed:
            fallback_tasks = [
                task for task in tasks if task.task_id in {result.task_id for result in failed}
            ]
            try:
                fallback_raw = self.dispatch_cpu(fallback_tasks)
                fallback_results = self._normalize_results(
                    fallback_tasks, fallback_raw, "cpu"
                )
            except BaseException as error:
                fallback_results = [
                    ConversionResult(
                        task.task_id,
                        False,
                        "cpu",
                        type(error).__name__,
                        fallback=True,
                    )
                    for task in fallback_tasks
                ]
            fallback_by_id = {result.task_id: result for result in fallback_results}
            results = [
                (
                    ConversionResult(
                        fallback_by_id[result.task_id].task_id,
                        fallback_by_id[result.task_id].ok,
                        fallback_by_id[result.task_id].backend,
                        fallback_by_id[result.task_id].error,
                        fallback=True,
                    )
                    if result.task_id in fallback_by_id
                    else result
                )
                if not result.ok
                else result
                for result in results
            ]

        for result in results:
            self._record_result(result)
        duration_ms = (time.perf_counter() - started) * 1000.0
        if self.metrics is not None and hasattr(self.metrics, "record_batch"):
            self.metrics.record_batch(
                backend,
                len(tasks),
                duration_ms=duration_ms,
                status="fallback" if failed and backend == "gpu" else "completed",
            )

    @staticmethod
    def _normalize_results(
        tasks: Sequence[ConversionTask], raw_results: Any, backend: str
    ) -> List[ConversionResult]:
        if isinstance(raw_results, bool):
            return [
                ConversionResult(task.task_id, raw_results, backend)
                for task in tasks
            ]
        if isinstance(raw_results, Mapping):
            values = [raw_results.get(task.task_id, False) for task in tasks]
        elif isinstance(raw_results, Iterable) and not isinstance(raw_results, (str, bytes)):
            values = list(raw_results)
        else:
            values = [False] * len(tasks)
        normalized: List[ConversionResult] = []
        for task, value in zip(tasks, values):
            if isinstance(value, ConversionResult):
                normalized.append(value)
            elif isinstance(value, tuple):
                ok = bool(value[0])
                error = str(value[1]) if len(value) > 1 and value[1] else None
                normalized.append(ConversionResult(task.task_id, ok, backend, error))
            else:
                normalized.append(ConversionResult(task.task_id, bool(value), backend))
        while len(normalized) < len(tasks):
            task = tasks[len(normalized)]
            normalized.append(
                ConversionResult(task.task_id, False, backend, "missing_result")
            )
        return normalized

    def _record_result(self, result: ConversionResult) -> None:
        with self._results_lock:
            self._results[result.task_id] = result
        if self.metrics is not None and hasattr(self.metrics, "increment"):
            self.metrics.increment(
                f"conversion_{'success' if result.ok else 'failure'}"
            )
            if result.fallback:
                self.metrics.increment("conversion_cpu_fallback")

    def _record(self, task: ConversionTask, ok: bool, backend: str, error: str) -> None:
        self._record_result(ConversionResult(task.task_id, ok, backend, error))

    def _record_queue(self, work_queue: queue.Queue[Any], name: str) -> None:
        if self.metrics is not None and hasattr(self.metrics, "record_queue"):
            self.metrics.record_queue(name, work_queue.qsize(), self.queue_size)
