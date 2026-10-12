import asyncio
import random

import pytest

from tinyserve.bench.loadgen import (
    Record,
    WorkItem,
    make_workload,
    percentile,
    run_load,
    summarize,
)
from tinyserve.bench.serve import serve_in_thread
from tinyserve.engine.engine import Engine
from tinyserve.server.app import create_app


def test_workload_is_deterministic_and_bounded():
    ids = list(range(5000))
    a = make_workload(ids, str, n=50, seed=3)
    b = make_workload(ids, str, n=50, seed=3)
    assert a == b
    assert all(8 <= w.prompt_tokens <= 400 and 16 <= w.max_tokens <= 256 for w in a)
    assert make_workload(ids, str, n=50, seed=4) != a


def test_poisson_gaps_have_unit_mean():
    gaps = [w.gap for w in make_workload(list(range(5000)), str, n=4000, seed=0)]
    assert sum(gaps) / len(gaps) == pytest.approx(1.0, rel=0.05)


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile(list(range(101)), 99) == 99


def test_summarize_known_numbers():
    recs = [
        Record(0, 200, t_send=0.0, t_first=0.5, t_end=2.5, completion_tokens=5),  # tpot 0.5
        Record(1, 200, t_send=1.0, t_first=1.2, t_end=3.0, completion_tokens=10),  # tpot 0.2
        Record(2, 429, t_send=1.5, error="queue full"),
    ]
    s = summarize(recs)
    assert (s["ok"], s["rejected_429"], s["errors"]) == (2, 1, 0)
    assert s["duration_s"] == 3.0 and s["output_tokens"] == 15
    assert s["throughput_tok_s"] == pytest.approx(5.0)
    assert s["ttft_mean_s"] == pytest.approx(0.35)
    assert s["tpot_p50_s"] == pytest.approx(0.35)


@pytest.mark.slow
def test_loadgen_against_live_server(model, tokenizer):
    rng = random.Random(0)
    items = [
        WorkItem("The quick brown fox " * rng.randint(1, 5), 0, rng.randint(5, 12), 0.05)
        for _ in range(8)
    ]
    engine = Engine(model, max_slots=4, max_len=256, decode=tokenizer.decode)
    with serve_in_thread(create_app(engine, tokenizer), port=8770) as url:
        records, wall = asyncio.run(run_load(url, items, rate=20))
    s = summarize(records)
    assert s["ok"] == 8 and s["errors"] == 0
    # ignore_eos: every request produced exactly its max_tokens
    assert [r.completion_tokens for r in records] == [w.max_tokens for w in items]
    assert all(r.t_send < r.t_first <= r.t_end for r in records)
    assert s["throughput_tok_s"] > 0 and wall > 0
