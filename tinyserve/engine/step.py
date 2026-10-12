"""Forward passes over the slot cache: prefill one request, or decode one token for many."""

import torch
from torch import Tensor

from tinyserve.engine.kv_cache import SlotKVCache
from tinyserve.model.gpt2 import GPT2


@torch.no_grad()
def prefill(model: GPT2, cache: SlotKVCache, slot: int, prompt_ids: list[int]) -> Tensor:
    """Run the whole prompt into `slot`; return next-token logits [vocab]."""
    view = cache.view(torch.tensor([slot]), T=len(prompt_ids))
    ids = torch.tensor([prompt_ids])
    return model(ids, view, view.positions(), view.key_mask())[0, -1]


@torch.no_grad()
def decode_step(model: GPT2, cache: SlotKVCache, slots: Tensor, tokens: Tensor) -> Tensor:
    """One new token per sequence. slots [B], tokens [B] -> logits [B, vocab].

    Sequences can be at different lengths: each row gets its own position and a key mask
    that hides the unused tail of shorter slots.
    """
    view = cache.view(slots, T=1)
    return model(tokens[:, None], view, view.positions(), view.key_mask())[:, -1]
