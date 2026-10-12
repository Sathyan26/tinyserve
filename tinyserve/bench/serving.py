"""Step 4.4: serial vs static vs continuous under Poisson load. The headline graph.

For each mode one server is started; it gets a warmup, then for every request rate the same
workload is replayed `runs` times. Results are saved after every run, so an interrupted
session (e.g. Colab) keeps everything measured so far.
"""

import asyncio
import json
import statistics
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from tinyserve.bench.loadgen import default_workload, run_load, summarize
from tinyserve.bench.plots import INK_2, SERIES, footer, hardware, power_plan, style
from tinyserve.bench.serve import serve_in_thread
from tinyserve.engine.engine import Engine
from tinyserve.model.gpt2 import GPT2
from tinyserve.server.app import create_app

MODES = ("serial", "static", "continuous")
DEFAULT_RATES = {"cpu": [0.1, 0.2, 0.35, 0.5, 0.75], "cuda": [0.5, 1, 2, 4, 8, 16]}
SLO_TTFT_S = 1.0  # latency target for the headline: interactive chat wants ~1 s to first token


def _wait_idle(engine: Engine, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while True:
        s = engine.stats()
        if not (s.queue_depth or s.waiting or s.running):
            return
        if time.monotonic() > deadline:
            raise RuntimeError("engine did not drain")
        time.sleep(0.1)


def measure(
    model: GPT2,
    tokenizer,
    out: Path,
    device: str,
    rates: list[float],
    runs: int,
    n: int,
    slots: int,
    modes: tuple[str, ...] = MODES,
    port: int = 8790,
) -> dict:
    items = default_workload(max(n, 10))
    data = {
        "config": {
            "model": "gpt2",
            "device": device,
            "hardware": hardware(),
            "power_plan": power_plan(),
            "dtype": str(model.dtype).replace("torch.", ""),
            "slots": slots,
            "requests_per_run": n,
            "runs": runs,
            "rates": rates,
            "date": date.today().isoformat(),
            "warmup": "10 requests per mode, discarded",
        },
        "rows": [],
    }
    for mode in modes:
        engine = Engine(
            model,
            max_slots=slots,
            max_len=1024,
            max_queue=10_000,
            request_timeout_s=None,
            decode=tokenizer.decode,
            mode=mode,
        )
        with serve_in_thread(create_app(engine, tokenizer), port=port) as url:
            asyncio.run(run_load(url, items[:10], rate=50))  # warmup
            _wait_idle(engine)
            for rate in rates:
                for run in range(runs):
                    engine.scheduler.batch_sizes.clear()
                    records, _ = asyncio.run(run_load(url, items[:n], rate))
                    _wait_idle(engine)
                    bs = list(engine.scheduler.batch_sizes)
                    s = summarize(records)
                    s["mean_batch"] = statistics.fmean(bs) if bs else 0.0
                    data["rows"].append({"mode": mode, "rate": rate, "run": run, **s})
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(data, indent=1))
                    print(
                        f"{mode:10s} rate {rate:5.2f}  run {run}  "
                        f"{s.get('throughput_tok_s', 0):6.1f} tok/s  "
                        f"TTFT p99 {s.get('ttft_p99_s', float('nan')):6.2f}s  "
                        f"batch {s['mean_batch']:.2f}  ok {s['ok']}/{s['requests']}",
                        flush=True,
                    )
    return data


def _median_by(rows: list[dict], mode: str, key: str) -> tuple[list, list, list, list]:
    rates = sorted({r["rate"] for r in rows if r["mode"] == mode})
    med, lo, hi = [], [], []
    for rate in rates:
        vals = [r[key] for r in rows if r["mode"] == mode and r["rate"] == rate and key in r]
        med.append(statistics.median(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return rates, med, lo, hi


def plot(data: dict, out: Path) -> None:
    style()
    rows, cfg = data["rows"], data["config"]
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    colors = {"serial": SERIES[1], "static": SERIES[3], "continuous": SERIES[0]}
    panels = [
        ("throughput_tok_s", "Throughput (output tokens/s)", False),
        ("ttft_p99_s", "p99 time to first token (s, log)", True),
        ("tpot_p50_s", "Median time per output token (s)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (key, title, log) in zip(axes, panels, strict=True):
        for mode in modes:
            x, med, lo, hi = _median_by(rows, mode, key)
            ax.errorbar(
                x,
                med,
                yerr=[
                    [m - a for m, a in zip(med, lo, strict=True)],
                    [b - m for m, b in zip(med, hi, strict=True)],
                ],
                marker="o",
                color=colors[mode],
                label=mode,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_xlabel("Request rate (req/s, Poisson)")
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    axes[1].axhline(SLO_TTFT_S, color=INK_2, linewidth=1, linestyle="--")
    axes[1].annotate(
        f"{SLO_TTFT_S:g} s target",
        (min(cfg["rates"]), SLO_TTFT_S),
        xytext=(0, 4),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )

    def slo_rate(mode: str) -> float | None:
        """Highest measured rate whose median p99 TTFT meets the target."""
        x, med, _, _ = _median_by(rows, mode, "ttft_p99_s")
        ok = [r for r, v in zip(x, med, strict=True) if v <= SLO_TTFT_S]
        return max(ok) if ok else None

    def peak(mode: str) -> float:
        return max(_median_by(rows, mode, "throughput_tok_s")[1])

    meets = {m: slo_rate(m) for m in modes}
    lowest = min(cfg["rates"])
    misses = [m for m in modes if meets[m] is None]
    cont = meets.get("continuous")
    slo_text = (
        f"under a {SLO_TTFT_S:g} s p99 TTFT target it sustains {cont:g} req/s"
        if cont is not None
        else f"no mode meets a {SLO_TTFT_S:g} s p99 TTFT target"
    )
    if misses:
        slo_text += f", while {' and '.join(misses)} miss it even at {lowest:g} req/s"
    ratios = ", ".join(
        f"{peak('continuous') / peak(m):.2f}x {m}" for m in modes if m != "continuous"
    )
    fig.suptitle(
        f"Continuous batching: {slo_text}.\nPeak throughput: {ratios}.",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    note = footer(
        f"GPT-2 small ({cfg['dtype']})",
        f"{cfg['slots']} slots, {cfg['requests_per_run']} req/run",
        f"median, min-max of {cfg['runs']} runs",
    )
    if cfg.get("power_plan") not in (None, "n/a", "High performance", "Ultimate Performance"):
        note += f" · power plan '{cfg['power_plan']}': preliminary"
    fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
