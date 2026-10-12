import pytest
import torch

from tests.conftest import PROMPTS
from tinyserve.engine.generate import generate
from tinyserve.engine.kv_cache import SlotKVCache
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.step import decode_step, prefill


def small_cache(max_slots: int = 3, max_len: int = 8) -> SlotKVCache:
    return SlotKVCache(n_layer=1, max_slots=max_slots, n_head=2, head_dim=4, max_len=max_len)


def test_alloc_free_reuse_resets_length():
    c = small_cache()
    a, b = c.alloc(), c.alloc()
    assert (a, b) == (0, 1) and c.num_free == 1
    c.lengths[a] = 5
    c.free(a)
    assert c.alloc() == a and c.lengths[a] == 0  # reused slot starts empty


def test_alloc_when_full_and_double_free_raise():
    c = small_cache(max_slots=1)
    s = c.alloc()
    with pytest.raises(RuntimeError):
        c.alloc()
    c.free(s)
    with pytest.raises(RuntimeError):
        c.free(s)


def test_decode_writes_each_row_at_its_own_position():
    c = small_cache(max_slots=6)
    c.lengths[torch.tensor([0, 5])] = torch.tensor([2, 4])
    view = c.view(torch.tensor([0, 5]), T=1)
    k = torch.stack([torch.full((2, 1, 4), 1.0), torch.full((2, 1, 4), 2.0)])  # [B, H, T, D]
    hk, _ = view.update(0, k, k)
    assert hk.shape == (2, 2, 5, 4)  # L = longest row (4) + 1
    assert c.k[0, 0, :, 2].eq(1).all() and c.k[0, 5, :, 4].eq(2).all()
    assert view.key_mask().tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    view.advance(1)
    assert c.lengths[[0, 5]].tolist() == [3, 5]


def run_lockstep(model, prompts, n, slots, max_slots=8):
    """Prefill each prompt into its slot, then greedy-decode all of them together."""
    cfg = model.cfg
    cache = SlotKVCache(cfg.n_layer, max_slots, cfg.n_head, cfg.head_dim, 128)
    for s in slots:  # claim exactly these slot ids
        cache._free.remove(s)
    first = [int(prefill(model, cache, s, p).argmax()) for s, p in zip(slots, prompts, strict=True)]
    outs = [[t] for t in first]
    slot_t = torch.tensor(slots)
    for _ in range(n - 1):
        logits = decode_step(model, cache, slot_t, torch.tensor([o[-1] for o in outs]))
        for o, row in zip(outs, logits, strict=True):
            o.append(int(row.argmax()))
    return outs


@pytest.mark.slow
def test_single_slot_matches_generate(model, tokenizer):
    ids = tokenizer.encode(PROMPTS[1])
    expect = generate(model, ids, SamplingParams(max_tokens=30, temperature=0)).token_ids
    assert run_lockstep(model, [ids], 30, slots=[3]) == [expect]


@pytest.mark.slow
def test_two_slots_different_lengths_match_alone(model, tokenizer):
    prompts = [tokenizer.encode(PROMPTS[0]), tokenizer.encode(PROMPTS[4])]  # 1 vs ~50 tokens
    p = SamplingParams(max_tokens=30, temperature=0)
    expect = [generate(model, ids, p).token_ids for ids in prompts]
    assert run_lockstep(model, prompts, 30, slots=[0, 5]) == expect
