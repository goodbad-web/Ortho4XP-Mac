import threading
import multiprocessing
import queue
import O4_UI_Utils as UI

################################################################################
class parallel_worker(threading.Thread):
    def __init__(self, task, queue, progress=None, success=[1]):
        threading.Thread.__init__(self)
        self._task = task
        self._queue = queue
        self._progress = progress
        self._success = success

    def run(self):
        while True:
            args = self._queue.get()
            if isinstance(args, str) and args == "quit":
                try:
                    UI.progress_bar(self._progress["bar"], 100)
                except:
                    pass
                return 1
            self._success[0] = self._task(*args) and self._success[0]
            if self._progress:
                self._progress["done"] += 1
                UI.progress_bar(
                    self._progress["bar"],
                    int(
                        100
                        * self._progress["done"]
                        / (self._progress["done"] + self._queue.qsize())
                    ),
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
    return success[0]

################################################################################
# Multiprocessing support
################################################################################
def multiprocessing_pool(task, arg_list, nbr_workers, progress=None, init_func=None, init_args=None):
    # This is a synchronous call but it uses a Pool to run in parallel.
    # It updates the progress bar.
    if not arg_list:
        return
    
    # Use spawn context explicitly for macOS stability
    ctx = multiprocessing.get_context("spawn")
    
    with ctx.Pool(processes=nbr_workers, initializer=init_func, initargs=(init_args,) if init_args else ()) as pool:
        done = 0
        total = len(arg_list)
        try:
            for _ in pool.imap_unordered(task_wrapper, [(task, args) for args in arg_list]):
                done += 1
                if progress:
                    progress["done"] += 1
                    UI.progress_bar(progress["bar"], int(100 * done / total))
                if UI.red_flag:
                    pool.terminate()
                    break
        except Exception as e:
            print(f"Pool execution error: {e}")
            pool.terminate()

def task_wrapper(args):
    task, task_args = args
    try:
        return task(*task_args)
    except Exception as e:
        print(f"Multiprocessing error: {e}")
        return 0