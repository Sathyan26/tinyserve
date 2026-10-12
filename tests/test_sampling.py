import pytest
import torch

from tests.conftest import PROMPTS
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams, apply_top_k, apply_top_p

# ---- fast unit tests: no model needed ----


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    kept = torch.isfinite(apply_top_k(logits, 2)).nonzero().flatten().tolist()
    assert kept == [1, 3]


def test_top_p_keeps_smallest_set_reaching_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    kept = torch.isfinite(apply_top_p(probs.log(), 0.75)).nonzero().flatten().tolist()
    assert kept == [0, 1]  # 0.5 < 0.75, 0.5 + 0.3 >= 0.75


def test_top_p_always_keeps_one_token():
    probs = torch.tensor([0.9, 0.1])
    assert torch.isfinite(apply_top_p(probs.log(), 0.01)).sum() == 1


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)


# ---- against the real model ----

N = 20


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_reference(model, ref_model, tokenizer, prompt):
    ids = tokenizer.encode(prompt)
    mine = generate(model, ids, SamplingParams(max_tokens=N, temperature=0)).token_ids
    ref = ref_model.generate(
        torch.tensor([ids]), max_new_tokens=N, do_sample=False, pad_token_id=50256
    )[0, len(ids) :].tolist()
    assert mine == ref[: len(mine)]


@pytest.mark.slow
def test_greedy_with_repetition_penalty_matches_reference(model, ref_model, tokenizer):
    ids = tokenizer.encode(PROMPTS[1])
    mine = generate(
        model, ids, SamplingParams(max_tokens=N, temperature=0, repetition_penalty=1.3)
    ).token_ids
    ref = ref_model.generate(
        torch.tensor([ids]),
        max_new_tokens=N,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=50256,
    )[0, len(ids) :].tolist()
    assert mine == ref[: len(mine)]


@pytest.mark.slow
def test_same_seed_same_tokens(model, tokenizer):
    ids = tokenizer.encode(PROMPTS[1])
    p = SamplingParams(max_tokens=N, temperature=0.8, top_p=0.9, top_k=50, seed=42)
    assert generate(model, ids, p).token_ids == generate(model, ids, p).token_ids


@pytest.mark.slow
def test_different_seed_different_tokens(model, tokenizer):
    ids = tokenizer.encode(PROMPTS[1])
    a = generate(model, ids, SamplingParams(max_tokens=N, temperature=1.0, seed=1))
    b = generate(model, ids, SamplingParams(max_tokens=N, temperature=1.0, seed=2))
    assert a.token_ids != b.token_ids


@pytest.mark.slow
def test_stop_sequence_trims_text(model, tokenizer):
    ids = tokenizer.encode("1, 2, 3, 4,")
    res = generate(
        model, ids, SamplingParams(max_tokens=30, temperature=0, stop=(" 7",)), tokenizer.decode
    )
    assert res.finish_reason == "stop"
    assert res.text == " 5, 6,"
