import random

import pytest

from tests.conftest import PROMPTS
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request, Scheduler

EXTRA = ["Once upon a time", "Q: What is 2 + 2?\nA:", "Dear Sir or Madam,", "import numpy as np\n"]


def make_workload(tokenizer, greedy: bool, n: int = 20, seed: int = 0):
    rng = random.Random(seed)
    texts = PROMPTS + EXTRA
    work = []
    for i in range(n):
        ids = tokenizer.encode(texts[i % len(texts)])
        max_tokens = rng.randint(5, 80)
        if greedy:
            p = SamplingParams(max_tokens=max_tokens, temperature=0)
        else:
            p = SamplingParams(
                max_tokens=max_tokens,
                temperature=rng.choice([0.6, 0.9, 1.2]),
                top_p=rng.choice([0.8, 0.95, 1.0]),
                top_k=rng.choice([0, 40]),
                repetition_penalty=rng.choice([1.0, 1.2]),
                seed=1000 + i,
            )
        work.append((ids, p))
    return work


@pytest.mark.slow
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_scheduler_matches_single_request(model, tokenizer, greedy):
    work = make_workload(tokenizer, greedy)
    sched = Scheduler(model, max_slots=4, max_len=256)
    reqs = [Request(ids, p) for ids, p in work]
    for r in reqs:
        sched.add(r)
    sched.run_until_done()

    # the run really exercised admission and eviction, not one big batch
    assert max(sched.batch_sizes) == 4
    assert sum(r.admitted_step > 0 for r in reqs) >= 10
    assert sched.cache.num_free == 4  # every slot came back

    for (ids, p), r in zip(work, reqs, strict=True):
        expect = generate(model, ids, p).token_ids
        assert r.result is not None and r.result.token_ids == expect, f"request {r.id} differs"


@pytest.mark.slow
def test_scheduler_stop_and_eos_free_slots_early(model, tokenizer):
    stop_req = Request(
        tokenizer.encode("1, 2, 3, 4,"), SamplingParams(max_tokens=50, temperature=0, stop=(" 7",))
    )
    long_req = Request(tokenizer.encode("Hello"), SamplingParams(max_tokens=40, temperature=0))
    sched = Scheduler(model, max_slots=2, max_len=128, decode=tokenizer.decode)
    sched.add(stop_req)
    sched.add(long_req)
    sched.run_until_done()
    assert stop_req.result.text == " 5, 6,"
    assert stop_req.finish_reason == "stop"
    assert stop_req.finish_time < long_req.finish_time  # left as soon as it was done


def test_rejects_oversized_request(model):
    sched = Scheduler(model, max_slots=1, max_len=16)
    with pytest.raises(ValueError):
        sched.add(Request([1] * 10, SamplingParams(max_tokens=10)))


# ---- 4.3: baseline modes share everything but the admission rule ----


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["serial", "static"])
def test_baseline_modes_match_single_request(model, tokenizer, mode):
    work = make_workload(tokenizer, greedy=True, n=10, seed=1)
    sched = Scheduler(model, max_slots=4, max_len=256, mode=mode, batch_wait_s=0)
    reqs = [Request(ids, p) for ids, p in work]
    for r in reqs:
        sched.add(r)
    sched.run_until_done()
    for (ids, p), r in zip(work, reqs, strict=True):
        assert r.out == generate(model, ids, p).token_ids

    if mode == "serial":
        assert max(sched.batch_sizes) == 1
    else:  # static: a batch is admitted together and nobody joins until all of it is done
        batches: dict[int, list[Request]] = {}
        for r in reqs:
            batches.setdefault(r.admitted_step, []).append(r)
        starts = sorted(batches)
        assert len(batches[starts[0]]) == 4  # full first batch
        for prev, nxt in zip(starts, starts[1:], strict=False):
            assert nxt >= max(r.finished_step for r in batches[prev])


def test_unknown_mode_rejected(model):
    with pytest.raises(ValueError):
        Scheduler(model, mode="fastest")


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_ignore_eos(model, monkeypatch, ignore_eos):
    """Force the sampler to always pick EOS: normally the request ends at once with no
    tokens; with ignore_eos it runs to exactly max_tokens (benchmarks rely on this)."""
    import tinyserve.engine.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "sample", lambda *a, **k: model.cfg.eos_token_id)
    p = SamplingParams(max_tokens=5, temperature=0, ignore_eos=ignore_eos)
    sched = Scheduler(model, max_slots=1, max_len=64)
    r = Request([464, 886], p)
    sched.add(r)
    sched.run_until_done()
    if ignore_eos:
        assert len(r.out) == 5 and r.finish_reason == "length"
    else:
        assert r.out == [] and r.finish_reason == "stop"
