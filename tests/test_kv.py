import pytest
import torch

from tests.conftest import PROMPTS
from tinyserve.engine.generate import generate
from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.sampler import SamplingParams


def test_cache_returns_history_and_rejects_overflow():
    c = KVCache(n_layer=1, batch=1, n_head=2, head_dim=4, max_len=3)
    k = torch.ones(1, 2, 2, 4)
    hk, _ = c.update(0, k, k)
    assert hk.shape == (1, 2, 2, 4)
    c.length = 2
    with pytest.raises(RuntimeError):
        c.update(0, k, k)  # 2 + 2 > 3


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_cached_equals_uncached_greedy(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt)
    p = SamplingParams(max_tokens=50, temperature=0)
    assert generate(model, ids, p, use_cache=True).token_ids == generate(
        model, ids, p, use_cache=False
    ).token_ids


@pytest.mark.slow
def test_cached_equals_uncached_sampled(model, tokenizer):
    ids = tokenizer.encode(PROMPTS[1])
    p = SamplingParams(max_tokens=50, temperature=0.9, top_p=0.95, seed=7)
    assert generate(model, ids, p, use_cache=True).token_ids == generate(
        model, ids, p, use_cache=False
    ).token_ids


@pytest.mark.slow
def test_cached_logits_match_full_forward(model, tokenizer):
    """Stronger than token equality: the cached step's logits equal the full pass's last row."""
    ids = torch.tensor([tokenizer.encode(PROMPTS[4])])
    cfg = model.cfg
    cache = KVCache(cfg.n_layer, 1, cfg.n_head, cfg.head_dim, ids.size(1))
    with torch.no_grad():
        model(ids[:, :-1], cache)  # prefill all but the last token
        step = model(ids[:, -1:], cache)[0, -1]
        full = model(ids)[0, -1]
    assert torch.allclose(step, full, atol=1e-4)
