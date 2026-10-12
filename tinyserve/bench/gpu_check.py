"""Correctness on the benchmark device, before any benchmark runs (step 4.6).

The test suite runs on CPU; this re-proves the key equivalences wherever the numbers will
be measured. Exit code 1 on any failure, so a notebook cell stops the run.

    python -m tinyserve.bench.gpu_check --device cuda
"""

import argparse
import sys

import torch

from tinyserve.engine.batch import generate_batch
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request, Scheduler
from tinyserve.model.registry import MODELS, load

PROMPTS = ["Hello", "The capital of France is", "def add(a, b):\n    return"]
LOGIT_ATOL = 1e-3  # GPU kernels (SDPA flash / efficient paths) sum in other orders than CPU


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def explained_by_near_tie(model, prompt: list[int], a: list[int], b: list[int]) -> str | None:
    """If a and b first differ at a step where the two candidates' logits are within 2 units
    in the last place (ulp) of the model's dtype, return a description; else None.

    In fp16 GPT-2's logits sit near -130, where the representable step is 0.125: two tokens
    closer than that are tied by rounding, and batch shape decides the winner (LOG.md).
    """
    k = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if k is None:
        return None
    with torch.no_grad():
        ctx = torch.tensor([prompt + a[:k]], device=model.device)
        logits = model(ctx)[0, -1].float()
    la, lb = logits[a[k]].item(), logits[b[k]].item()
    mag = max(abs(la), abs(lb), 1e-6)
    ulp = torch.finfo(model.dtype).eps * 2 ** torch.floor(torch.log2(torch.tensor(mag))).item()
    if abs(la - lb) <= 2 * ulp:
        return f"near-tie at token {k}: gap {abs(la - lb):.4f} <= 2 ulp ({2 * ulp:.4f})"
    return None


def same_or_near_tie(model, prompts, got, alone, exact: bool) -> tuple[bool, str]:
    notes = []
    for prompt, g, a in zip(prompts, got, alone, strict=True):
        if g == a:
            continue
        why = None if exact else explained_by_near_tie(model, prompt, a, g)
        if why is None:
            return False, "tokens differ and it is NOT a near-tie"
        notes.append(why)
    return True, "; ".join(notes)


def run_model(model_name: str, device: str) -> bool:
    from transformers import AutoModelForCausalLM

    print(f"\n{model_name} on {device}", flush=True)
    ok = True
    model, tok, _ = load(model_name)
    model = model.to(device)
    ref = AutoModelForCausalLM.from_pretrained(MODELS[model_name].repo, dtype=torch.float32)
    ref = ref.to(device).eval()
    prompts = [tok.encode(p) for p in PROMPTS]

    with torch.no_grad():
        diffs = [
            (model(torch.tensor([p], device=device)) - ref(torch.tensor([p], device=device)).logits)
            .abs()
            .max()
            .item()
            for p in prompts
        ]
    ok &= check("logits vs HF (fp32)", max(diffs) < LOGIT_ATOL, f"max |diff| {max(diffs):.1e}")

    for dtype in (torch.float32, torch.float16):
        m = model.to(dtype)  # nn.Module.to() converts in place: fp32 must run first
        label = str(dtype).replace("torch.", "")
        p = SamplingParams(max_tokens=16, temperature=0)
        alone = [generate(m, ids, p).token_ids for ids in prompts]
        batched = [r.token_ids for r in generate_batch(m, prompts, p)[0]]
        sched = Scheduler(m, max_slots=2, max_len=128)
        reqs = [Request(ids, p) for ids in prompts]
        for r in reqs:
            sched.add(r)
        sched.run_until_done()
        exact = dtype == torch.float32  # fp32: must be identical; fp16: identical or near-tie
        for what, got in (("padded batch", batched), ("scheduler", [r.out for r in reqs])):
            same, note = same_or_near_tie(m, prompts, got, alone, exact)
            ok &= check(f"{label}: {what} == alone", same, note)
        ok &= check(f"{label}: KV cache on {device}", sched.cache.k.device.type == device)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", nargs="+", default=sorted(MODELS))
    args = ap.parse_args()
    print(f"torch {torch.__version__}", flush=True)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    results = [run_model(name, args.device) for name in args.models]
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
