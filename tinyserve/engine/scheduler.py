"""Continuous batching: one decode step at a time for every running request.

Requests join between steps as soon as a slot is free and leave the moment they finish,
so a short request never waits for a long one to end (unlike static batching).
"""

import itertools
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tinyserve.engine.generate import GenerationResult, finalize, find_stop
from tinyserve.engine.kv_cache import SlotKVCache
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
from tinyserve.engine.step import decode_step, prefill
from tinyserve.model.gpt2 import GPT2

_ids = itertools.count()


@dataclass(eq=False)
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    # called with (token_id or None, finish_reason or None) from the engine thread
    on_token: Callable[[int | None, str | None], None] | None = None
    id: int = field(default_factory=lambda: next(_ids))

    # filled in by the scheduler
    slot: int | None = None
    out: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    result: GenerationResult | None = None
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None
    admitted_step: int | None = None
    _gen: torch.Generator | None = None
    _seen: Tensor | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class Scheduler:
    def __init__(
        self,
        model: GPT2,
        max_slots: int = 8,
        max_len: int | None = None,
        decode: Callable[[list[int]], str] | None = None,
    ) -> None:
        cfg = model.cfg
        self.model = model
        self.decode = decode
        self.max_len = min(max_len or cfg.n_ctx, cfg.n_ctx)
        self.cache = SlotKVCache(cfg.n_layer, max_slots, cfg.n_head, cfg.head_dim, self.max_len)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.done: list[Request] = []
        self.steps = 0
        self.batch_sizes: list[int] = []  # decode batch size per step, for metrics

    def add(self, req: Request) -> None:
        if len(req.prompt_ids) + req.params.max_tokens > self.max_len:
            raise ValueError(f"prompt + max_tokens exceeds max_len {self.max_len}")
        if req.params.stop and self.decode is None:
            raise ValueError("stop sequences need a decode function")
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> None:
        """ADMIT waiting requests into free slots, then DECODE one token for all running."""
        self._admit()
        if self.running:
            slots = torch.tensor([r.slot for r in self.running])
            last = torch.tensor([r.out[-1] for r in self.running])
            logits = decode_step(self.model, self.cache, slots, last)
            self.batch_sizes.append(len(self.running))
            for req, row in zip(self.running, logits, strict=True):
                self._accept(req, row)
            self._evict()
        self.steps += 1

    def run_until_done(self) -> list[Request]:
        while self.has_work:
            self.step()
        return self.done

    def _admit(self) -> None:
        while self.waiting and self.cache.num_free:
            req = self.waiting.popleft()
            req.slot = self.cache.alloc()
            req.admitted_step = self.steps
            req._gen = make_generator(req.params)
            req._seen = torch.tensor(req.prompt_ids)
            logits = prefill(self.model, self.cache, req.slot, req.prompt_ids)
            self.running.append(req)
            self._accept(req, logits)  # prefill's output is the first generated token
        self._evict()  # a request can finish on its first token (EOS or max_tokens=1)

    def _accept(self, req: Request, logits: Tensor) -> None:
        """Sample one token for `req` and decide whether it is finished.

        Mirrors generate() exactly; the equivalence tests depend on that.
        """
        assert req._gen is not None and req._seen is not None
        tok = sample(logits, req._seen, req.params, req._gen)
        now = time.monotonic()
        if req.first_token_time is None:
            req.first_token_time = now
        if tok == self.model.cfg.eos_token_id:
            req.finish_reason = "stop"
        else:
            req.out.append(tok)
            req._seen = torch.cat([req._seen, torch.tensor([tok])])
            if req.on_token:
                req.on_token(tok, None)
            if req.params.stop and find_stop(self.decode(req.out), req.params.stop) is not None:  # type: ignore[misc]
                req.finish_reason = "stop"
            elif len(req.out) >= req.params.max_tokens:
                req.finish_reason = "length"
        if req.finished:
            req.finish_time = now

    def _evict(self) -> None:
        still = []
        for req in self.running:
            if not req.finished:
                still.append(req)
                continue
            assert req.slot is not None
            self.cache.free(req.slot)  # slot is reusable on the very next admit
            req.slot = None
            req.result = finalize(req.out, req.finish_reason or "length", req.params, self.decode)
            if req.on_token:
                req.on_token(None, req.finish_reason)
            self.done.append(req)
        self.running = still
