"""Baseline floor: naive Hugging Face `generate()`, one request after another.

This is what serving looks like with no engine at all: no batching, no shared cache, each
request waits for the previous one. transformers is used here ONLY as the baseline being
compared against (dev dependency), never inside tinyserve.

    uv run python -m tinyserve.bench.hf_floor --n 30
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tinyserve.bench.loadgen import default_workload
from tinyserve.bench.plots import hardware, power_plan

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def measure(n: int, runs: int, device: str) -> dict:
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
    items = default_workload(n)

    def one(item) -> int:
        # same chat template the server applies, so prompts are the same length
        ids = tok(f"User: {item.prompt}\nAssistant:", return_tensors="pt").input_ids.to(device)
        out = model.generate(
            ids,
            max_new_tokens=item.max_tokens,
            min_new_tokens=item.max_tokens,  # = ignore_eos: exactly max_tokens, like tinyserve
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        return out.size(1) - ids.size(1)

    one(items[0])  # warmup
    rates = []
    for _ in range(runs):
        t = time.perf_counter()
        tokens = sum(one(item) for item in items)
        rates.append(tokens / (time.perf_counter() - t))
        print(f"HF generate(), serial: {rates[-1]:.1f} tok/s", flush=True)
    return {
        "config": {
            "hardware": hardware(),
            "power_plan": power_plan(),
            "device": device,
            "requests": n,
            "runs": runs,
        },
        "throughput_tok_s": statistics.median(rates),
        "throughput_runs": rates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    data = measure(args.n, args.runs, args.device)
    out = ROOT / "docs" / "results" / args.device / "gpt2" / "hf_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"median {data['throughput_tok_s']:.1f} tok/s -> {out}")


if __name__ == "__main__":
    main()
