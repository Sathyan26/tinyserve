"""One command regenerates every graph in the README.

    uv run python -m tinyserve.bench.run_all                 # measure everything (slow)
    uv run python -m tinyserve.bench.run_all --plot-only     # redraw from saved results
    uv run python -m tinyserve.bench.run_all --only serving --quick   # pipeline smoke test

Raw results go to docs/results/<device>/<model>/, graphs to docs/img/.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from tinyserve.bench import serving

ROOT = Path(__file__).resolve().parents[2]
STEPS = ("kv", "quant", "serving")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate every README graph.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", choices=STEPS, action="append", help="run only these steps")
    ap.add_argument("--plot-only", action="store_true", help="redraw from saved JSON")
    ap.add_argument("--quick", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--rates", type=float, nargs="+", help="request rates (req/s)")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--n", type=int, default=30, help="requests per run")
    ap.add_argument("--slots", type=int, default=8)
    args = ap.parse_args()
    steps = args.only or list(STEPS)
    results = ROOT / "docs" / "results" / args.device / "gpt2"
    img = ROOT / "docs" / "img"

    # kv and quant have their own scripts; run them as modules so each stays standalone
    for step, module in (("kv", "kv_graph"), ("quant", "quant_graph")):
        if step in steps:
            extra = ["--plot-only"] if args.plot_only else []
            subprocess.run([sys.executable, "-m", f"tinyserve.bench.{module}", *extra], check=True)

    if "serving" in steps:
        out_json = results / ("serving_quick.json" if args.quick else "serving.json")
        out_png = img / ("headline_quick.png" if args.quick else "headline.png")
        if args.plot_only:
            data = json.loads(out_json.read_text())
        else:
            from tinyserve.model.gpt2 import GPT2
            from tinyserve.model.tokenizer import Tokenizer
            from tinyserve.model.weights import download

            folder = download()
            model = GPT2.from_pretrained(folder).to(args.device)
            rates = args.rates or ([0.5] if args.quick else serving.DEFAULT_RATES[args.device])
            runs = args.runs or (1 if args.quick else 3 if args.device == "cpu" else 5)
            n = 8 if args.quick else args.n
            data = serving.measure(
                model, Tokenizer(folder), out_json, args.device, rates, runs, n, args.slots
            )
        serving.plot(data, out_png)


if __name__ == "__main__":
    main()
