"""Step 1.7: generation time vs sequence length, with and without the KV cache.

uv run python -m tinyserve.bench.kv_graph
"""

import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from tinyserve.bench.plots import INK_2, SERIES, footer, style
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.model.gpt2 import GPT2
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import download

LENGTHS = [32, 64, 128, 256, 512]
RUNS = 3
PROMPT = "The history of computing began"
ROOT = Path(__file__).resolve().parents[2]


def time_generation(model: GPT2, ids: list[int], n: int, use_cache: bool) -> float:
    # temperature 0 + no EOS stop: GPT-2 greedy rarely emits EOS, and we assert the length below
    p = SamplingParams(max_tokens=n, temperature=0)
    t0 = time.perf_counter()
    res = generate(model, ids, p, use_cache=use_cache)
    elapsed = time.perf_counter() - t0
    assert len(res.token_ids) == n, "hit EOS early; pick another prompt"
    return elapsed


def measure() -> dict:
    folder = download()
    model = GPT2.from_pretrained(folder)
    ids = Tokenizer(folder).encode(PROMPT)
    for use_cache in (True, False):  # warmup: first calls pay allocator / thread-pool setup
        time_generation(model, ids, 16, use_cache)

    rows = []
    for n in LENGTHS:
        for use_cache in (True, False):
            times = [time_generation(model, ids, n, use_cache) for _ in range(RUNS)]
            row = {
                "tokens": n,
                "cache": use_cache,
                "median_s": statistics.median(times),
                "min_s": min(times),
                "max_s": max(times),
            }
            rows.append(row)
            print(f"{n:4d} tokens  cache={use_cache!s:5}  {row['median_s']:.2f}s")
    return {"prompt_tokens": len(ids), "runs": RUNS, "rows": rows}


def plot(data: dict, out: Path) -> None:
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    series = [(False, "No cache", SERIES[1]), (True, "KV cache", SERIES[0])]
    for cache, label, color in series:
        rows = [r for r in data["rows"] if r["cache"] is cache]
        n = [r["tokens"] for r in rows]
        total = [r["median_s"] for r in rows]
        lo = [r["median_s"] - r["min_s"] for r in rows]
        hi = [r["max_s"] - r["median_s"] for r in rows]
        per_tok = [1000 * t / k for t, k in zip(total, n, strict=True)]
        left.errorbar(n, total, yerr=[lo, hi], marker="o", color=color, label=label, capsize=3)
        right.plot(n, per_tok, marker="o", color=color, label=label)
        left.annotate(
            f"{label}  {total[-1]:.1f}s",
            (n[-1], total[-1]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color=INK_2,
        )
        right.annotate(
            f"{per_tok[-1]:.0f} ms",
            (n[-1], per_tok[-1]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color=INK_2,
        )

    no = next(r for r in data["rows"] if r["tokens"] == LENGTHS[-1] and not r["cache"])
    yes = next(r for r in data["rows"] if r["tokens"] == LENGTHS[-1] and r["cache"])
    fig.suptitle(
        f"KV cache: {no['median_s'] / yes['median_s']:.1f}x faster at {LENGTHS[-1]} tokens; "
        "per-token cost stops growing with length",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    left.set_title("Total generation time")
    left.set_xlabel("Generated tokens")
    left.set_ylabel("Seconds (median)")
    right.set_title("Time per token")
    right.set_xlabel("Generated tokens")
    right.set_ylabel("Milliseconds per token")
    for ax in (left, right):
        ax.set_xticks(LENGTHS)
        ax.set_ylim(bottom=0)
    right.legend(loc="upper left")
    fig.text(
        0.01, 0.01, footer("GPT-2 small", "fp32", f"median of {data['runs']}"), fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys

    results = ROOT / "docs" / "results" / "cpu" / "gpt2" / "kv_cache.json"
    if "--plot-only" in sys.argv:  # redraw from saved results, no re-measuring
        data = json.loads(results.read_text())
    else:
        data = measure()
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text(json.dumps(data, indent=2))
    plot(data, ROOT / "docs" / "img" / "kv_cache.png")
