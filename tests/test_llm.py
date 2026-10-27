import json
import threading
import time

import httpx

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


def test_hosted_embed_adds_prefix_sends_key_and_keeps_order(monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.headers["authorization"], body["input"]))
        # Returned out of order: embed() must sort by index.
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i)]}
                                                  for i in reversed(range(len(body["input"])))]})

    real_client = httpx.Client
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    cfg = CFG.model_copy(update={"provider": CFG.provider.model_copy(update={
        "embed_name": "openai", "embed_url": "https://example.test/v1", "embed_api_key_env": "TEST_EMBED_KEY",
        "embed_prefix": "p: "})})
    assert llm.embed(["a", "b", "c"], cfg, batch_size=2) == [[0.0], [1.0], [0.0]]
    assert seen == [("Bearer k", ["p: a", "p: b"]), ("Bearer k", ["p: c"])]
