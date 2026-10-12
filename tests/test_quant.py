import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.conftest import PROMPTS
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request, Scheduler
from tinyserve.quant.int8 import QuantLinear, cast, model_bytes, quantize_int8

# ---- QuantLinear on its own ----


def test_quantlinear_error_bounded_by_half_scale():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    q = QuantLinear(lin)
    assert q.w_int8.dtype == torch.int8 and q.scale.shape == (32,)
    err = (q.dequantized() - lin.weight).abs()
    assert (err <= q.scale[:, None] / 2 + 1e-7).all()  # rounding error per weight


def test_quantlinear_output_close_to_float():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128)
    x = torch.randn(4, 256)
    rel = (QuantLinear(lin)(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01


def test_quantize_selected_blocks_only(model):
    q = quantize_int8(model, layers=[0, 5])
    quantized = [isinstance(b.mlp.c_fc, QuantLinear) for b in q.h]
    assert quantized == [i in (0, 5) for i in range(12)]
    assert isinstance(model.h[0].mlp.c_fc, nn.Linear)  # original untouched (copy by default)


def test_int8_memory(model):
    # 85M block weights go 4 bytes -> 1; the 39M tied embedding + positions stay fp32
    assert model_bytes(quantize_int8(model)) / model_bytes(model) < 0.5


# ---- quality against fp32 (thresholds ~5-7x above what was measured; see LOG.md) ----


def kl_and_agreement(model, other, tokenizer):
    kls, agree = [], []
    with torch.no_grad():
        for p in PROMPTS:
            ids = torch.tensor([tokenizer.encode(p)])
            a, b = model(ids)[0].float(), other(ids)[0].float()
            log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
            kls.append(F.kl_div(log_b, log_a, log_target=True, reduction="batchmean"))
            agree.append((a.argmax(-1) == b.argmax(-1)).float().mean())
    return float(torch.stack(kls).mean()), float(torch.stack(agree).mean())


@pytest.mark.slow
@pytest.mark.parametrize(
    "variant, max_kl, min_agree",
    [("int8", 0.01, 0.95), ("fp16", 0.002, 0.98), ("bf16", 0.06, 0.85)],
)
def test_quantized_quality(model, tokenizer, variant, max_kl, min_agree):
    other = {
        "int8": lambda: quantize_int8(model),
        "fp16": lambda: cast(model, torch.float16),
        "bf16": lambda: cast(model, torch.bfloat16),
    }[variant]()
    kl, agree = kl_and_agreement(model, other, tokenizer)
    assert kl < max_kl, f"KL {kl:.5f}"
    assert agree > min_agree, f"top-1 agreement {agree:.3f}"


# ---- quantized models run through the whole engine ----


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["int8", "fp16"])
def test_scheduler_runs_quantized_model(model, tokenizer, variant):
    m = quantize_int8(model) if variant == "int8" else cast(model, torch.float16)
    p = SamplingParams(max_tokens=20, temperature=0)
    prompts = [tokenizer.encode(t) for t in PROMPTS[:3]]
    sched = Scheduler(m, max_slots=2, max_len=128)
    reqs = [Request(ids, p) for ids in prompts]
    for r in reqs:
        sched.add(r)
    sched.run_until_done()
    assert sched.cache.k.dtype == m.dtype
    for ids, r in zip(prompts, reqs, strict=True):
        assert r.out == generate(m, ids, p).token_ids  # batching invariance holds per dtype
