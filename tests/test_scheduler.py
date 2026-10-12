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
