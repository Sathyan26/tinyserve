import pytest
import torch

from tests.conftest import PROMPTS
from tinyserve.engine.batch import generate_batch, left_pad
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams

EXTRA = [
    "Once upon a time",
    "Q: What is 2 + 2?\nA:",
    "The quick brown fox jumps over the lazy dog. " * 4,
]
N = 30


def test_left_pad_right_aligns_and_masks():
    ids, mask = left_pad([[5], [1, 2, 3]], pad_id=0)
    assert ids.tolist() == [[0, 0, 5], [1, 2, 3]]
    assert mask.tolist() == [[False, False, True], [True, True, True]]
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    assert pos.tolist() == [[0, 0, 0], [0, 1, 2]]  # padded row's real token is at position 0


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    ids = [tokenizer.encode(p) for p in PROMPTS + EXTRA]
    assert min(map(len, ids)) == 1 and max(map(len, ids)) >= 40  # really mixed lengths
    return ids


@pytest.fixture(scope="module")
def alone_greedy(model, prompt_ids):
    p = SamplingParams(max_tokens=N, temperature=0)
    return [generate(model, ids, p).token_ids for ids in prompt_ids]


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_batch_matches_alone_greedy(model, prompt_ids, alone_greedy, batch_size):
    p = SamplingParams(max_tokens=N, temperature=0)
    for start in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[start : start + batch_size]
        results, _ = generate_batch(model, chunk, p)
        for i, res in enumerate(results):
            assert res.token_ids == alone_greedy[start + i], f"prompt {start + i} differs"


@pytest.mark.slow
def test_batch_matches_alone_seeded_mixed_params(model, prompt_ids):
    """Every row has its own sampling settings and seed; batch mates must not matter."""
    params = [
        SamplingParams(max_tokens=5 + 4 * i, temperature=0.7 + 0.1 * i, top_p=0.9, seed=i)
        for i in range(len(prompt_ids))
    ]
    batched, stats = generate_batch(model, prompt_ids, params)
    for ids, p, res in zip(prompt_ids, params, batched, strict=True):
        assert res.token_ids == generate(model, ids, p).token_ids
    assert stats.wasted_row_steps > 0  # short requests idled while the longest ran


@pytest.mark.slow
def test_padded_prefill_has_no_nan(model, prompt_ids):
    ids, mask = left_pad(prompt_ids, pad_id=50256)
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model(ids, positions=pos, key_mask=mask)
    assert torch.isfinite(logits).all()
