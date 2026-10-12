"""Load generator: Poisson arrivals, realistic lengths, streaming client, latency metrics.

The workload (prompts, output lengths, arrival gaps) is generated once from a seed and saved,
so every mode, rate and server sees exactly the same requests. Gaps are stored for a rate of
1 req/s and scaled by 1/rate, so changing the rate changes only the spacing.

    uv run python -m tinyserve.bench.loadgen --rate 2 --n 50
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = ROOT / "docs" / "results" / "workload.json"


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    prompt_tokens: int
    max_tokens: int
    gap: float  # seconds before this request at 1 req/s (unit-rate exponential)


def lognormal_int(rng: random.Random, median: float, sigma: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(rng.lognormvariate(math.log(median), sigma))))


def make_workload(
    text_ids: list[int],
    decode: Callable[[list[int]], str],
    n: int = 200,
    seed: int = 0,
    prompt_median: int = 100,
    output_median: int = 64,
) -> list[WorkItem]:
    """Prompts are real WikiText passages; both lengths are right-skewed (lognormal), as
    production traffic is: many short requests, a few long ones."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        p_len = lognormal_int(rng, prompt_median, 0.6, 8, 400)
        start = rng.randrange(len(text_ids) - p_len)
        items.append(
            WorkItem(
                prompt=decode(text_ids[start : start + p_len]),
                prompt_tokens=p_len,
                max_tokens=lognormal_int(rng, output_median, 0.7, 16, 256),
                gap=rng.expovariate(1.0),
            )
        )
    return items


def save_workload(items: list[WorkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(i) for i in items], indent=1))


def load_workload(path: Path) -> list[WorkItem]:
    return [WorkItem(**d) for d in json.loads(path.read_text())]


def default_workload(n: int = 200) -> list[WorkItem]:
    """Load the saved workload, creating it (deterministically) on first use."""
    if DEFAULT_WORKLOAD.exists():
        items = load_workload(DEFAULT_WORKLOAD)
        if len(items) >= n:
            return items[:n]
    from tinyserve.bench.perplexity import load_wikitext2_test
    from tinyserve.model.tokenizer import Tokenizer
    from tinyserve.model.weights import download

    tok = Tokenizer(download())
    items = make_workload(tok.encode(load_wikitext2_test()), tok.decode, n=max(n, 200))
    save_workload(items, DEFAULT_WORKLOAD)
    return items[:n]


@dataclass
class Record:
    index: int
    status: int = 0
    t_send: float = 0.0
    t_first: float | None = None  # first non-empty content chunk
    t_end: float | None = None
    completion_tokens: int = 0
    error: str | None = None


async def _one(
    client: httpx.AsyncClient, url: str, i: int, item: WorkItem, send_at: float, model: str
) -> Record:
    await asyncio.sleep(max(0.0, send_at - time.perf_counter()))
    rec = Record(index=i, t_send=time.perf_counter())
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item.prompt}],
        "max_tokens": item.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # every request produces exactly max_tokens
    }
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            rec.status = r.status_code
            if r.status_code != 200:
                rec.error = (await r.aread()).decode()[:200]
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    rec.error = chunk["error"]["message"]
                elif chunk.get("usage"):
                    rec.completion_tokens = chunk["usage"]["completion_tokens"]
                elif rec.t_first is None and chunk["choices"][0]["delta"].get("content"):
                    rec.t_first = time.perf_counter()
        rec.t_end = time.perf_counter()
    except httpx.HTTPError as e:
        rec.error = f"{type(e).__name__}: {e}"
    return rec


async def run_load(
    url: str, items: list[WorkItem], rate: float, model: str = "gpt2", timeout: float = 600
) -> tuple[list[Record], float]:
    """Send every item at its Poisson arrival time; return records and wall-clock seconds."""
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        offsets, t = [], 0.0
        for item in items:
            t += item.gap / rate
            offsets.append(t)
        tasks = [
            _one(client, url, i, item, t0 + off, model)
            for i, (item, off) in enumerate(zip(items, offsets, strict=True))
        ]
        records = await asyncio.gather(*tasks)
    return list(records), time.perf_counter() - t0


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(records: list[Record]) -> dict:
    ok = [r for r in records if r.status == 200 and r.error is None and r.t_end is not None]
    out: dict = {
        "requests": len(records),
        "ok": len(ok),
        "rejected_429": sum(r.status == 429 for r in records),
        "errors": sum(
            r.status != 429 and (r.error is not None or r.status != 200) for r in records
        ),
    }
    if not ok:
        return out
    start = min(r.t_send for r in ok)
    end = max(r.t_end for r in ok)  # type: ignore[type-var]
    tokens = sum(r.completion_tokens for r in ok)
    ttft = [r.t_first - r.t_send for r in ok if r.t_first is not None]
    e2e = [r.t_end - r.t_send for r in ok]  # type: ignore[operator]
    tpot = [
        (r.t_end - r.t_first) / (r.completion_tokens - 1)  # type: ignore[operator]
        for r in ok
        if r.t_first is not None and r.completion_tokens > 1
    ]
    out |= {
        "duration_s": end - start,
        "output_tokens": tokens,
        "throughput_tok_s": tokens / (end - start),
        "requests_per_s": len(ok) / (end - start),
    }
    for name, vals in (("ttft", ttft), ("tpot", tpot), ("e2e", e2e)):
        out[f"{name}_mean_s"] = statistics.fmean(vals) if vals else float("nan")
        out[f"{name}_p50_s"] = percentile(vals, 50)
        out[f"{name}_p99_s"] = percentile(vals, 99)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rate", type=float, default=1.0, help="mean requests per second")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    records, _ = asyncio.run(run_load(args.url, default_workload(args.n), args.rate, args.model))
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
