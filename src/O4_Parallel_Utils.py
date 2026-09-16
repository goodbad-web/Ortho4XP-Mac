import threading
import multiprocessing
import queue
import O4_UI_Utils as UI

################################################################################
class parallel_worker(threading.Thread):
    def __init__(self, task, queue, progress=None, success=None):
        threading.Thread.__init__(self)
        self._task = task
        self._queue = queue
        self._progress = progress
        self._success = success
        self.success = True

    def run(self):
        while True:
            args = self._queue.get()
            if isinstance(args, str) and args == "quit":
                try:
                    UI.progress_bar(
                        self._progress["bar"],
                        100,
                        self._progress.get("message"),
                    )
                except:
                    pass
                return self.success
            try:
                task_success = bool(self._task(*args))
            except Exception as error:
                task_success = False
                UI.vprint(0, "ERROR: Parallel task failed:", error)
            self.success = self.success and task_success
            if self._success is not None:
                self._success[0] = int(bool(self._success[0]) and task_success)
            if self._progress:
                self._progress["done"] += 1
                UI.progress_bar(
                    self._progress["bar"],
                    int(
                        100
                        * self._progress["done"]
                        / (self._progress["done"] + self._queue.qsize())
                    ),
                    self._progress.get("message"),
                )
            if UI.red_flag:
                return 0

################################################################################
def parallel_launch(task, queue, nbr_workers, progress=None):
    workers = []
    for _ in range(nbr_workers):
        worker = parallel_worker(task, queue, progress)
        worker.start()
        workers.append(worker)
    return workers

################################################################################
def parallel_join(workers):
    for worker in workers:
        worker.join()
    return int(bool(workers) and all(worker.success for worker in workers) and not UI.red_flag)

################################################################################
def parallel_execute(task, execute_queue, nbr_workers, progress=None):
    success = [1]
    for _ in range(nbr_workers):
        execute_queue.put("quit")
    workers = []
    for _ in range(nbr_workers):
        worker = parallel_worker(task, execute_queue, progress, success)
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return int(bool(success[0]) and all(worker.success for worker in workers) and not UI.red_flag)

################################################################################
# Multiprocessing support
################################################################################
class ReusableMultiprocessingPool:
    """Keep one macOS spawn pool alive across bounded conversion batches."""

    def __init__(self, nbr_workers, init_func=None, init_args=None):
        if nbr_workers < 1:
            raise ValueError("nbr_workers must be positive")
        ctx = multiprocessing.get_context("spawn")
        initargs = (init_args,) if init_args is not None else ()
        self._pool = ctx.Pool(
            processes=nbr_workers,
            initializer=init_func,
            initargs=initargs,
        )
        self._closed = False

    def run(self, task, arg_list, progress=None):
        if not arg_list:
            return 1
        if self._closed:
            raise RuntimeError("reusable multiprocessing pool is closed")
        done = 0
        success = 0
        total = len(arg_list)
        log_step = max(1, total // 10)
        try:
            for res in self._pool.imap_unordered(
                task_wrapper, [(task, args) for args in arg_list]
            ):
                done += 1
                if res:
                    success += 1
                if progress:
                    progress["done"] += 1
                    UI.progress_bar(
                        progress["bar"],
                        int(100 * done / total),
                        progress.get("message"),
                    )
                if done % log_step == 0 or done == total:
                    UI.vprint(1, f"   ... {done}/{total} ({int(100 * done / total)}%)")
                if UI.red_flag:
                    self.terminate()
                    break
        except Exception as e:
            UI.vprint(1, f"Pool execution error: {e}")
            self.terminate()
        return int(done == total and success == total and not UI.red_flag)

    def close(self):
        if not self._closed:
            self._pool.close()
            self._pool.join()
            self._closed = True

    def terminate(self):
        if not self._closed:
            self._pool.terminate()
            self._pool.join()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.terminate()
        return False


def multiprocessing_pool(task, arg_list, nbr_workers, progress=None, init_func=None, init_args=None):
    # Preserve the historical one-shot API for callers outside the streaming
    # scheduler while sharing the same implementation and spawn semantics.
    if not arg_list:
        return 1
    with ReusableMultiprocessingPool(
        nbr_workers,
        init_func=init_func,
        init_args=init_args,
    ) as pool:
        return pool.run(task, arg_list, progress=progress)

def task_wrapper(args):
    task, task_args = args
    try:
        return task(*task_args)
    except Exception as e:
        print(f"Multiprocessing error: {e}")
        return 0
