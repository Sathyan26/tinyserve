"""Single-request generation: the obviously-correct path every faster path is tested against."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
from tinyserve.model.gpt2 import GPT2


@dataclass
class GenerationResult:
    token_ids: list[int]
    finish_reason: str  # "length" | "stop" (EOS or stop string)
    text: str | None = None


def find_stop(text: str, stops: tuple[str, ...]) -> int | None:
    """Index where the earliest stop string begins, or None."""
    hits = [i for s in stops if (i := text.find(s)) != -1]
    return min(hits) if hits else None


@torch.no_grad()
def generate(
    model: GPT2,
    prompt_ids: list[int],
    params: SamplingParams,
    decode: Callable[[list[int]], str] | None = None,
    use_cache: bool = True,
) -> GenerationResult:
    """use_cache=False re-runs the whole sequence every step (the reference path).

    use_cache=True: prefill the prompt once, then decode feeding one token per step.
    """
    if params.stop and decode is None:
        raise ValueError("stop sequences need a decode function")
    cfg = model.cfg
    if len(prompt_ids) + params.max_tokens > cfg.n_ctx:
        raise ValueError(f"prompt + max_tokens exceeds context window {cfg.n_ctx}")

    g = make_generator(params)
    ids = torch.tensor(prompt_ids)
    cache = None
    if use_cache:
        max_len = len(prompt_ids) + params.max_tokens
        cache = KVCache(cfg.n_layer, 1, cfg.n_head, cfg.head_dim, max_len)

    out: list[int] = []
    finish = "length"
    feed: Tensor = ids  # prefill feeds the whole prompt; decode feeds only the new token
    for _ in range(params.max_tokens):
        logits = model(feed[None], cache)[0, -1]
        tok = sample(logits, ids, params, g)
        if tok == cfg.eos_token_id:
            finish = "stop"
            break
        out.append(tok)
        ids = torch.cat([ids, torch.tensor([tok])])
        feed = ids[-1:] if use_cache else ids
        if params.stop and find_stop(decode(out), params.stop) is not None:  # type: ignore[misc]
            finish = "stop"
            break

    return finalize(out, finish, params, decode)


def finalize(
    out: list[int],
    finish: str,
    params: SamplingParams,
    decode: Callable[[list[int]], str] | None,
) -> GenerationResult:
    text = decode(out) if decode else None
    if text is not None and finish == "stop" and (cut := find_stop(text, params.stop)) is not None:
        text = text[:cut]  # stop string itself is not returned (OpenAI behaviour)
    return GenerationResult(out, finish, text)
