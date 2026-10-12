"""Static batching: run several prompts as one padded batch until the longest finishes."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tinyserve.engine.generate import GenerationResult, finalize, find_stop
from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
from tinyserve.model.gpt2 import GPT2


@dataclass
class BatchStats:
    steps: int = 0  # forward passes after prefill
    row_steps: int = 0  # rows computed across those steps
    wasted_row_steps: int = 0  # rows computed for sequences that had already finished

    @property
    def wasted_fraction(self) -> float:
        return self.wasted_row_steps / self.row_steps if self.row_steps else 0.0


def left_pad(prompts: Sequence[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align every prompt so all last tokens share the final column.

    Returns ids [B, L] and mask [B, L] (True = real token).
    """
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), L, dtype=torch.bool)
    for b, p in enumerate(prompts):
        ids[b, L - len(p) :] = torch.tensor(p)
        mask[b, L - len(p) :] = True
    return ids, mask


@torch.no_grad()
def generate_batch(
    model: GPT2,
    prompts: Sequence[list[int]],
    params: SamplingParams | Sequence[SamplingParams],
    decode: Callable[[list[int]], str] | None = None,
) -> tuple[list[GenerationResult], BatchStats]:
    cfg = model.cfg
    B = len(prompts)
    plist = list(params) if isinstance(params, Sequence) else [params] * B
    if any(p.stop for p in plist) and decode is None:
        raise ValueError("stop sequences need a decode function")

    ids, key_mask = left_pad(prompts, pad_id=cfg.eos_token_ids[0])
    # position = how many real tokens came before, NOT the column index. With arange,
    # a prompt padded by 3 would start at position 3 and silently change its output.
    positions = (key_mask.long().cumsum(-1) - 1).clamp(min=0)

    max_new = max(p.max_tokens for p in plist)
    cache = KVCache(
        cfg.n_layer,
        B,
        cfg.n_kv_head,
        cfg.head_dim,
        ids.size(1) + max_new,
        dtype=model.dtype,
        device=model.device,
    )
    gens = [make_generator(p) for p in plist]
    seen = [torch.tensor(p) for p in prompts]  # real tokens only: padding never penalised
    outs: list[list[int]] = [[] for _ in range(B)]
    finish = ["length"] * B
    done = [False] * B
    stats = BatchStats()

    feed = ids
    for step in range(max_new):
        dev = model.device
        logits = model(feed.to(dev), cache, positions.to(dev), key_mask.to(dev))[:, -1]
        logits = logits.float().cpu()
        if step > 0:
            stats.steps += 1
            stats.row_steps += B
            stats.wasted_row_steps += sum(done)
        nxt = []
        for b in range(B):
            if done[b]:
                nxt.append(cfg.eos_token_ids[0])  # filler; its output is never read
                continue
            tok = sample(logits[b], seen[b], plist[b], gens[b])
            if tok in cfg.eos_token_ids and not plist[b].ignore_eos:
                done[b], finish[b] = True, "stop"
            else:
                outs[b].append(tok)
                seen[b] = torch.cat([seen[b], torch.tensor([tok])])
                if plist[b].stop and find_stop(decode(outs[b]), plist[b].stop) is not None:  # type: ignore[misc]
                    done[b], finish[b] = True, "stop"
                elif len(outs[b]) >= plist[b].max_tokens:
                    done[b] = True
            nxt.append(tok)
        if all(done):
            break
        feed = torch.tensor(nxt)[:, None]
        positions = positions[:, -1:] + 1
        key_mask = torch.cat([key_mask, torch.ones(B, 1, dtype=torch.bool)], dim=1)

    results = [finalize(outs[b], finish[b], plist[b], decode) for b in range(B)]
    return results, stats
