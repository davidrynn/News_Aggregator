import threading
import time

from newsagg import config, llm

CFG = config.load()


def with_concurrency(n: int) -> config.Config:
    return CFG.model_copy(update={"provider": CFG.provider.model_copy(update={"concurrency": n})})


def test_parallel_keeps_item_order_when_later_items_finish_first():
    def slow_first(n: int) -> int:
        time.sleep(0.05 if n == 0 else 0)
        return n * 10
    assert llm.parallel(slow_first, range(6), with_concurrency(4)) == [0, 10, 20, 30, 40, 50]


def test_parallel_overlaps_calls_up_to_concurrency():
    running, peak, lock = 0, 0, threading.Lock()

    def call(_):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
    llm.parallel(call, range(8), with_concurrency(3))
    assert peak == 3


def test_concurrency_1_runs_in_the_calling_thread():
    main = threading.get_ident()
    assert llm.parallel(lambda _: threading.get_ident() == main, range(3), with_concurrency(1)) == [True] * 3
