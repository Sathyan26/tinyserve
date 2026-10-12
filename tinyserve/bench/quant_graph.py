"""Step 4.2: memory vs speed vs quality for fp32 / fp16 / bf16 / int8.

uv run python -m tinyserve.bench.quant_graph [--plot-only]
"""

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tinyserve.bench.perplexity import load_wikitext2_test, perplexity
from tinyserve.bench.plots import INK_2, SERIES, footer, style
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.model.gpt2 import GPT2
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import download
from tinyserve.quant.int8 import cast, model_bytes, quantize_int8

PPL_TOKENS = 50_000
GEN_TOKENS = 128
RUNS = 5
PROMPT = "The history of computing began"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "cpu" / "gpt2" / "quant.json"


def variants(base: GPT2) -> dict[str, GPT2]:
    return {
        "fp32": base,
        "fp16": cast(base, torch.float16),
        "bf16": cast(base, torch.bfloat16),
        "int8": quantize_int8(base),
    }


def tokens_per_second(models: dict[str, GPT2], ids: list[int]) -> dict[str, list[float]]:
    """Round-robin: one run of every variant per round, so CPU clock drift (heat, power
    state) lands on all variants equally instead of on whichever ran last."""
    p = SamplingParams(max_tokens=GEN_TOKENS, temperature=0)
    for m in models.values():  # warmup
        generate(m, ids, SamplingParams(max_tokens=16, temperature=0))
    rates: dict[str, list[float]] = {name: [] for name in models}
    for _ in range(RUNS):
        for name, m in models.items():
            t = time.perf_counter()
            n = len(generate(m, ids, p).token_ids)
            rates[name].append(n / (time.perf_counter() - t))
    return rates


def measure(reuse_ppl: dict[str, float] | None = None) -> dict:
    """reuse_ppl: perplexities from a previous run (deterministic, and the slow part)."""
    folder = download()
    tok = Tokenizer(folder)
    text_ids = tok.encode(load_wikitext2_test())[:PPL_TOKENS]
    models = variants(GPT2.from_pretrained(folder))
    rates = tokens_per_second(models, tok.encode(PROMPT))
    rows = []
    for name, m in models.items():
        ppl = (reuse_ppl or {}).get(name) or perplexity(lambda c, m=m: m(c), text_ids)
        r = rates[name]
        rows.append(
            {
                "dtype": name,
                "mb": model_bytes(m) / 1e6,
                "tok_s": statistics.median(r),
                "tok_s_min": min(r),
                "tok_s_max": max(r),
                "tok_s_runs": r,
                "ppl": ppl,
            }
        )
        print(
            f"{name:5s} {rows[-1]['mb']:6.0f} MB  {rows[-1]['tok_s']:5.1f} tok/s "
            f"[{min(r):.1f}-{max(r):.1f}]  ppl {ppl:.3f}"
        )
    return {"ppl_tokens": len(text_ids), "gen_tokens": GEN_TOKENS, "runs": RUNS, "rows": rows}


def plot(data: dict, out: Path) -> None:
    style()
    rows = data["rows"]
    names = [r["dtype"] for r in rows]
    base = next(r for r in rows if r["dtype"] == "fp32")
    y = list(range(len(rows)))[::-1]  # fp32 on top
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    colors = SERIES[: len(rows)]

    a1.barh(y, [r["mb"] for r in rows], color=colors, height=0.6)
    a1.set_title("Weight memory (MB)")
    a2.barh(
        y,
        [r["tok_s"] for r in rows],
        color=colors,
        height=0.6,
        xerr=[
            [r["tok_s"] - r["tok_s_min"] for r in rows],
            [r["tok_s_max"] - r["tok_s"] for r in rows],
        ],
        error_kw={"ecolor": INK_2, "capsize": 3, "elinewidth": 1},
    )
    a2.set_title(f"Tokens/s (median of {data['runs']})")
    dppl = [100 * (r["ppl"] / base["ppl"] - 1) for r in rows]
    a3.barh(y, dppl, color=colors, height=0.6)
    a3.set_title("Perplexity vs fp32 (%)")
    a3.axvline(0, color=INK_2, linewidth=0.8)

    # labels sit past the end of the bar OR its whisker, whichever is further right
    for ax, vals, ends, fmt in (
        (a1, [r["mb"] for r in rows], [r["mb"] for r in rows], "{:.0f}"),
        (a2, [r["tok_s"] for r in rows], [r["tok_s_max"] for r in rows], "{:.1f}"),
        (a3, dppl, dppl, "{:+.2f}"),
    ):
        span = max(abs(e) for e in ends)
        for yi, v, e in zip(y, vals, ends, strict=True):
            ax.text(e + span * 0.03, yi, fmt.format(v), va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, span * 1.3)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, names)

    fast16 = [r for r in rows if r["dtype"] in ("fp16", "bf16")]
    lo16 = min(r["tok_s"] for r in fast16) / base["tok_s"]
    hi16 = max(r["tok_s"] for r in fast16) / base["tok_s"]
    int8 = next(r for r in rows if r["dtype"] == "int8")
    bf16 = next(r for r in rows if r["dtype"] == "bf16")
    fig.suptitle(
        f"16-bit halves memory and decodes {lo16:.1f}-{hi16:.1f}x faster on CPU; "
        f"int8 keeps quality but is {base['tok_s'] / int8['tok_s']:.1f}x slower; "
        f"bf16 costs {100 * (bf16['ppl'] / base['ppl'] - 1):.1f}% perplexity",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        footer(
            "GPT-2 small",
            f"WikiText-2 ppl on {data['ppl_tokens'] // 1000}k tokens",
            f"{data['gen_tokens']} greedy tokens, round-robin",
        ),
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        data = json.loads(RESULTS.read_text())
    else:
        old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"rows": []}
        reuse = {r["dtype"]: r["ppl"] for r in old["rows"]} if "--reuse-ppl" in sys.argv else None
        data = measure(reuse)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(data, indent=2))
    plot(data, ROOT / "docs" / "img" / "quant.png")
