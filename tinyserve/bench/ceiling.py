"""Baseline ceiling: the same load generator and workload against an external server (vLLM).

Start vLLM separately (see notebooks/colab_benchmarks.ipynb), then:
    python -m tinyserve.bench.ceiling --url http://127.0.0.1:8001 \\
        --served-model openai-community/gpt2 --model gpt2 --device cuda
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from tinyserve.bench.loadgen import default_workload, run_load, summarize
from tinyserve.bench.plots import hardware
from tinyserve.bench.serving import DEFAULT_RATES

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--served-model", required=True, help="model name the server expects")
    ap.add_argument("--model", required=True, help="tinyserve name, for the results path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--server", default="vllm")
    ap.add_argument("--rates", type=float, nargs="+")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    items = default_workload(max(args.n, 10))
    rates = args.rates or DEFAULT_RATES[args.device]
    out = ROOT / "docs" / "results" / args.device / args.model / f"{args.server}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "config": {
            "server": args.server,
            "served_model": args.served_model,
            "hardware": hardware(),
            "requests_per_run": args.n,
            "runs": args.runs,
            "rates": rates,
            "date": date.today().isoformat(),
        },
        "rows": [],
    }
    asyncio.run(run_load(args.url, items[:10], rate=50, model=args.served_model))  # warmup
    for rate in rates:
        for run in range(args.runs):
            records, _ = asyncio.run(run_load(args.url, items[: args.n], rate, args.served_model))
            s = summarize(records)
            data["rows"].append({"mode": args.server, "rate": rate, "run": run, **s})
            out.write_text(json.dumps(data, indent=1))
            print(
                f"{args.server} rate {rate:5.2f} run {run}  "
                f"{s.get('throughput_tok_s', 0):7.1f} tok/s  "
                f"TTFT p99 {s.get('ttft_p99_s', float('nan')):6.2f}s  ok {s['ok']}/{s['requests']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
