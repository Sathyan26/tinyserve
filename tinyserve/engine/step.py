"""Forward passes over the slot cache: prefill one request, or decode one token for many.

Device rule: slot bookkeeping and sampling live on the CPU; only model inputs move to the
model's device, and logits come back to the CPU. Seeded sampling therefore uses the same
CPU generator on a laptop and on a GPU.
"""

import torch
from torch import Tensor

from tinyserve.engine.kv_cache import SlotKVCache
from tinyserve.model.gpt2 import GPT2


@torch.no_grad()
def prefill(model: GPT2, cache: SlotKVCache, slot: int, prompt_ids: list[int]) -> Tensor:
    """Run the whole prompt into `slot`; return next-token logits [vocab]."""
    dev = model.device
    view = cache.view(torch.tensor([slot]), T=len(prompt_ids))
    ids = torch.tensor([prompt_ids], device=dev)
    logits = model(ids, view, view.positions().to(dev), view.key_mask().to(dev))[0, -1]
    return logits.float().cpu()


@torch.no_grad()
def decode_step(model: GPT2, cache: SlotKVCache, slots: Tensor, tokens: Tensor) -> Tensor:
    """One new token per sequence. slots [B], tokens [B] -> logits [B, vocab].

    Sequences can be at different lengths: each row gets its own position and a key mask
    that hides the unused tail of shorter slots.
    """
    dev = model.device
    view = cache.view(slots, T=1)
    logits = model(
        tokens[:, None].to(dev), view, view.positions().to(dev), view.key_mask().to(dev)
    )[:, -1]
    return logits.float().cpu()
