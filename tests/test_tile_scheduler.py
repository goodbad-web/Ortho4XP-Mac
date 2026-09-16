import sys
import threading
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from O4_Tile_Scheduler import (  # noqa: E402
    ConversionResult,
    ConversionTask,
    TileConversionScheduler,
)


def test_gpu_failure_falls_back_only_failed_tasks():
    gpu_batches = []
    cpu_batches = []

    def dispatch_gpu(tasks):
        gpu_batches.append([task.task_id for task in tasks])
        return {task.task_id: task.task_id != "gpu-fails" for task in tasks}

    def dispatch_cpu(tasks):
        cpu_batches.append([task.task_id for task in tasks])
        return {task.task_id: True for task in tasks}

    scheduler = TileConversionScheduler(
        dispatch_cpu=dispatch_cpu,
        dispatch_gpu=dispatch_gpu,
        gpu_eligible=lambda task: task.metadata.get("gpu", False),
        queue_size=4,
        cpu_batch_size=2,
        gpu_batch_size=4,
        batch_wait_ms=0,
    )
    scheduler.submit(ConversionTask("gpu-ok", (), {"gpu": True}))
    scheduler.submit(ConversionTask("gpu-fails", (), {"gpu": True}))
    scheduler.submit(ConversionTask("cpu", (), {"gpu": False}))
    results = scheduler.wait()

    by_id = {result.task_id: result for result in results}
    assert gpu_batches == [["gpu-ok", "gpu-fails"]]
    assert cpu_batches == [["gpu-fails"], ["cpu"]] or cpu_batches == [["cpu"], ["gpu-fails"]]
    assert by_id["gpu-ok"].backend == "gpu"
    assert by_id["gpu-ok"].ok
    assert by_id["gpu-fails"].backend == "cpu"
    assert by_id["gpu-fails"].fallback
    assert by_id["gpu-fails"].ok
    assert by_id["cpu"].backend == "cpu"


def test_close_flushes_partial_batch_without_losing_tasks():
    batches = []

    def dispatch_cpu(tasks):
        batches.append([task.task_id for task in tasks])
        return True

    scheduler = TileConversionScheduler(
        dispatch_cpu=dispatch_cpu,
        queue_size=2,
        cpu_batch_size=8,
        batch_wait_ms=1000,
    )
    for index in range(3):
        assert scheduler.submit(ConversionTask(str(index), ()))
    results = scheduler.wait()

    assert {result.task_id for result in results} == {"0", "1", "2"}
    assert batches == [["0", "1", "2"]]


def test_cancel_marks_queued_work_and_returns():
    started = threading.Event()
    release = threading.Event()

    def dispatch_cpu(tasks):
        started.set()
        release.wait(1)
        return True

    scheduler = TileConversionScheduler(
        dispatch_cpu=dispatch_cpu,
        queue_size=2,
        cpu_batch_size=1,
        batch_wait_ms=0,
    )
    assert scheduler.submit(ConversionTask("first", ()))
    assert started.wait(1)
    assert scheduler.submit(ConversionTask("second", ()))
    scheduler.cancel()
    release.set()
    results = scheduler.wait(timeout=2)

    by_id = {result.task_id: result for result in results}
    assert by_id["first"].ok
    assert by_id["second"].backend == "cancelled"
    assert not by_id["second"].ok


def test_worker_failure_drains_accepted_work_without_hanging():
    def gpu_eligible(task):
        if task.task_id == "boom":
            raise RuntimeError("eligibility failure")
        return True

    scheduler = TileConversionScheduler(
        dispatch_cpu=lambda tasks: True,
        dispatch_gpu=lambda tasks: True,
        gpu_eligible=gpu_eligible,
        queue_size=4,
        batch_wait_ms=0,
    )
    for task_id in ("boom", "queued-1", "queued-2"):
        assert scheduler.submit(ConversionTask(task_id, ()))

    results = scheduler.wait(timeout=2)

    assert scheduler.error is not None
    assert scheduler.submitted_count == 3
    assert {result.task_id for result in results} == {
        "boom",
        "queued-1",
        "queued-2",
    }
    assert all(not result.ok for result in results)


def test_worker_failure_during_wait_does_not_join_forever():
    entered = threading.Event()
    release = threading.Event()
    allow_failure = threading.Event()
    wait_observation_armed = threading.Event()
    wait_checked = threading.Event()

    class ObservedScheduler(TileConversionScheduler):
        @property
        def error(self):
            value = super().error
            if wait_observation_armed.is_set():
                wait_checked.set()
            return value

    def dispatch_cpu(tasks):
        entered.set()
        release.wait()
        return True

    def gpu_eligible(task):
        if task.task_id == "boom":
            allow_failure.wait()
            raise RuntimeError("eligibility failure")
        return False

    scheduler = ObservedScheduler(
        dispatch_cpu=dispatch_cpu,
        dispatch_gpu=lambda tasks: True,
        gpu_eligible=gpu_eligible,
        batch_wait_ms=0,
    )
    assert scheduler.submit(ConversionTask("normal", ()))
    assert entered.wait(1)
    assert scheduler.submit(ConversionTask("boom", ()))
    scheduler.close()

    wait_errors = []
    wait_done = threading.Event()

    def wait_for_scheduler():
        try:
            scheduler.wait()
        except BaseException as error:
            wait_errors.append(error)
        finally:
            wait_done.set()

    wait_thread = threading.Thread(target=wait_for_scheduler, daemon=True)
    wait_observation_armed.set()
    wait_thread.start()
    try:
        assert wait_checked.wait(1)
        allow_failure.set()
        assert wait_done.wait(2)
        assert len(wait_errors) == 1
        assert isinstance(wait_errors[0], TimeoutError)
    finally:
        release.set()
        wait_thread.join(2)
