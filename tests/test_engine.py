import random
import threading
import time

import pytest

from tests.test_scheduler import make_workload
from tinyserve.engine.engine import Engine, QueueFull
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request


class Collector:
    """Gathers one request's streamed tokens; `done` fires on the finish callback."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.finish: str | None = None
        self.done = threading.Event()

    def __call__(self, tok: int | None, finish: str | None) -> None:
        if tok is not None:
            self.tokens.append(tok)
        if finish is not None:
            self.finish = finish
            self.done.set()


def submit_with_collector(engine: Engine, ids: list[int], p: SamplingParams, **kw):
    c = Collector()
    req = Request(ids, p, on_token=c, **kw)
    engine.submit(req)
    return req, c


# ---- 3.2: the centerpiece's acceptance test ----

N_REQ = 30


@pytest.fixture(scope="module")
def workload(tokenizer):
    work = make_workload(tokenizer, greedy=False, n=N_REQ, seed=3)
    # half greedy, half seeded sampling, max_tokens trimmed to keep CI time reasonable
    out = []
    for i, (ids, p) in enumerate(work):
        mt = 5 + p.max_tokens % 40
        if i % 2:
            p = SamplingParams(max_tokens=mt, temperature=0)
        else:
            p = SamplingParams(
                max_tokens=mt, temperature=p.temperature, top_p=p.top_p, seed=p.seed
            )
        out.append((ids, p))
    return out


@pytest.fixture(scope="module")
def expected(model, workload):
    return [generate(model, ids, p).token_ids for ids, p in workload]


@pytest.mark.slow
@pytest.mark.parametrize("arrival_seed", range(5))
def test_random_arrival_matches_single_request(model, workload, expected, arrival_seed):
    engine = Engine(model, max_slots=4, max_len=256, max_queue=N_REQ).start()
    rng = random.Random(arrival_seed)
    order = list(range(N_REQ))
    rng.shuffle(order)  # random arrival ORDER as well as random gaps
    collectors: dict[int, Collector] = {}

    def client() -> None:
        for i in order:
            time.sleep(rng.expovariate(20))  # Poisson arrivals, mean gap 50 ms
            _, collectors[i] = submit_with_collector(engine, *workload[i])

    t = threading.Thread(target=client)
    t.start()
    t.join()
    try:
        for i in order:
            assert collectors[i].done.wait(120), f"request {i} never finished"
            assert collectors[i].tokens == expected[i], f"request {i} differs"
    finally:
        engine.stop()
    assert engine.errors == 0


# ---- production behaviours ----


@pytest.mark.slow
def test_cancel_frees_slot(model, tokenizer):
    engine = Engine(model, max_slots=1, max_len=256).start()
    try:
        req, c = submit_with_collector(
            engine, tokenizer.encode("Hello"), SamplingParams(max_tokens=200, temperature=0)
        )
        while len(c.tokens) < 3:
            time.sleep(0.01)
        req.cancelled.set()
        assert c.done.wait(5)
        assert c.finish == "cancelled" and len(c.tokens) < 200
        # the single slot is usable again
        _, c2 = submit_with_collector(
            engine, tokenizer.encode("Hi"), SamplingParams(max_tokens=3, temperature=0)
        )
        assert c2.done.wait(30) and c2.finish == "length"
    finally:
        engine.stop()


@pytest.mark.slow
def test_timeout(model, tokenizer):
    engine = Engine(model, max_slots=1, max_len=256, request_timeout_s=0.3).start()
    try:
        _, c = submit_with_collector(
            engine, tokenizer.encode("Hello"), SamplingParams(max_tokens=200, temperature=0)
        )
        assert c.done.wait(10)
        assert c.finish == "timeout"
    finally:
        engine.stop()


def test_queue_full_rejects(model):
    engine = Engine(model, max_slots=1, max_queue=2)  # not started: nothing drains the queue
    p = SamplingParams(max_tokens=2)
    engine.submit(Request([1, 2], p))
    engine.submit(Request([1, 2], p))
    with pytest.raises(QueueFull):
        engine.submit(Request([1, 2], p))


def test_invalid_request_rejected_before_queueing(model):
    engine = Engine(model, max_len=16)
    with pytest.raises(ValueError):
        engine.submit(Request([1] * 10, SamplingParams(max_tokens=10)))
    assert engine.stats().queue_depth == 0


@pytest.mark.slow
def test_engine_survives_a_failing_step(model, tokenizer, monkeypatch):
    engine = Engine(model, max_slots=2, max_len=256)
    real_step = engine.scheduler.step
    calls = {"n": 0}

    def flaky_step() -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated kernel failure")
        real_step()

    monkeypatch.setattr(engine.scheduler, "step", flaky_step)
    engine.start()
    try:
        _, c1 = submit_with_collector(
            engine, tokenizer.encode("Hello"), SamplingParams(max_tokens=50, temperature=0)
        )
        assert c1.done.wait(30) and c1.finish == "error"
        _, c2 = submit_with_collector(
            engine, tokenizer.encode("Hi"), SamplingParams(max_tokens=3, temperature=0)
        )
        assert c2.done.wait(30) and c2.finish == "length"  # engine kept serving
        assert engine.errors == 1 and engine.stats().free_slots == 2
    finally:
        engine.stop()
