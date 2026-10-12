"""Continuous batching: one decode step at a time for every running request.

Requests join between steps as soon as a slot is free and leave the moment they finish,
so a short request never waits for a long one to end (unlike static batching).

The same loop also runs the two baselines it is benchmarked against, so every mode shares
the model code, the cache and the sampler; only the admission rule differs:
  continuous  admit into any free slot between any two steps
  static      admit a whole batch (max_slots requests, or whatever arrived within
              batch_wait_s), then admit nobody until the entire batch has finished
  serial      one request at a time (no batching at all)
"""

import itertools
import threading
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

MODES = ("continuous", "static", "serial")


@dataclass(eq=False)
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    # called with (token_id, None) per token, then (None, finish_reason) once, from the
    # engine thread; callers must hop to their own thread/loop before touching shared state
    on_token: Callable[[int | None, str | None], None] | None = None
    # perf_counter, not monotonic: monotonic() ticks only every 15 ms on Windows
    deadline: float | None = None  # time.perf_counter() after which the request is dropped
    id: int = field(default_factory=lambda: next(_ids))
    # set from any thread (e.g. the client disconnected); the scheduler drops it between steps
    cancelled: threading.Event = field(default_factory=threading.Event)

    # filled in by the scheduler
    slot: int | None = None
    out: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # length | stop | cancelled | timeout | error
    result: GenerationResult | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None
    admitted_step: int | None = None
    finished_step: int | None = None
    _gen: torch.Generator | None = None
    _seen: Tensor | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def abort_reason(self, now: float) -> str | None:
        if self.cancelled.is_set():
            return "cancelled"
        if self.deadline is not None and now > self.deadline:
            return "timeout"
        return None


class Scheduler:
    def __init__(
        self,
        model: GPT2,
        max_slots: int = 8,
        max_len: int | None = None,
        decode: Callable[[list[int]], str] | None = None,
        keep_done: bool = True,
        mode: str = "continuous",
        batch_wait_s: float = 0.2,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cfg = model.cfg
        self.mode = mode
        self.batch_wait_s = batch_wait_s  # static: how long a partial batch may wait
        self.model = model
        self.decode = decode
        self.max_len = min(max_len or cfg.n_ctx, cfg.n_ctx)
        self.cache = SlotKVCache(
            cfg.n_layer, max_slots, cfg.n_head, cfg.head_dim, self.max_len, dtype=model.dtype
        )
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.keep_done = keep_done  # a long-lived server must not keep every request forever
        self.done: list[Request] = []
        self.steps = 0
        self.batch_sizes: deque[int] = deque(maxlen=10_000)  # recent decode batch sizes
        # optional hooks (metrics); called on the engine thread
        self.on_step: Callable[[int], None] | None = None  # decode batch size
        self.on_finish: Callable[[Request], None] | None = None

    def validate(self, req: Request) -> None:
        """Raise ValueError for a request that can never run. Safe to call from any thread."""
        if not req.prompt_ids:
            raise ValueError("prompt is empty")
        if len(req.prompt_ids) + req.params.max_tokens > self.max_len:
            raise ValueError(
                f"prompt ({len(req.prompt_ids)} tokens) + max_tokens "
                f"({req.params.max_tokens}) exceeds max_len {self.max_len}"
            )
        if req.params.stop and self.decode is None:
            raise ValueError("stop sequences need a decode function")

    def add(self, req: Request) -> None:
        self.validate(req)
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> bool:
        """REAP aborted requests, ADMIT waiting ones, DECODE one token for all running.

        Returns False when there was nothing to decode (lets the engine thread sleep).
        """
        self._reap()
        self._admit()
        did_decode = bool(self.running)
        if self.running:
            slots = torch.tensor([r.slot for r in self.running])
            last = torch.tensor([r.out[-1] for r in self.running])
            logits = decode_step(self.model, self.cache, slots, last)
            self.batch_sizes.append(len(self.running))
            if self.on_step:
                self.on_step(len(self.running))
            for req, row in zip(self.running, logits, strict=True):
                self._accept(req, row)
            self._evict()
        self.steps += 1
        return did_decode

    def run_until_done(self) -> list[Request]:
        while self.has_work:
            self.step()
        return self.done

    def fail_all(self, reason: str = "error") -> None:
        """After an unexpected exception: end every request so no client waits forever."""
        for req in [*self.running, *self.waiting]:
            if not req.finished:
                req.finish_reason = reason
                req.finish_time = time.perf_counter()
        self.waiting.clear()
        self._evict()

    def _reap(self) -> None:
        now = time.perf_counter()
        for req in [*self.running, *self.waiting]:
            if not req.finished and (reason := req.abort_reason(now)):
                req.finish_reason = reason
                req.finish_time = now
        dropped = [r for r in self.waiting if r.finished]
        self.waiting = deque(r for r in self.waiting if not r.finished)
        for req in dropped:
            self._finish(req)
        self._evict()

    def _admission_budget(self) -> int:
        """How many waiting requests may be admitted right now, under this mode."""
        if self.mode == "continuous":
            return self.cache.num_free
        if self.running:  # static and serial: never join a batch in flight
            return 0
        if self.mode == "serial":
            return 1
        full = len(self.waiting) >= self.cache.num_free
        waited = self.waiting and (
            time.perf_counter() - self.waiting[0].arrival_time >= self.batch_wait_s
        )
        return self.cache.num_free if (full or waited) else 0

    def _admit(self) -> None:
        budget = self._admission_budget()
        while self.waiting and self.cache.num_free and budget > 0:
            budget -= 1
            req = self.waiting.popleft()
            req.slot = self.cache.alloc()
            req.admitted_step = self.steps
            req._gen = make_generator(req.params)
            req._seen = torch.tensor(req.prompt_ids)
            self.running.append(req)  # before prefill, so fail_all() can free its slot
            logits = prefill(self.model, self.cache, req.slot, req.prompt_ids)
            self._accept(req, logits)  # prefill's output is the first generated token
        self._evict()  # a request can finish on its first token (EOS or max_tokens=1)

    def _accept(self, req: Request, logits: Tensor) -> None:
        """Sample one token for `req` and decide whether it is finished.

        Mirrors generate() exactly; the equivalence tests depend on that.
        """
        assert req._gen is not None and req._seen is not None
        tok = sample(logits, req._seen, req.params, req._gen)
        now = time.perf_counter()
        if req.first_token_time is None:
            req.first_token_time = now
        if tok == self.model.cfg.eos_token_id and not req.params.ignore_eos:
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
            req.finished_step = self.steps

    def _evict(self) -> None:
        still = []
        for req in self.running:
            if not req.finished:
                still.append(req)
                continue
            assert req.slot is not None
            self.cache.free(req.slot)  # slot is reusable on the very next admit
            req.slot = None
            self._finish(req)
        self.running = still

    def _finish(self, req: Request) -> None:
        req.result = finalize(req.out, req.finish_reason or "error", req.params, self.decode)
        if req.on_token:
            req.on_token(None, req.finish_reason)
        if self.on_finish:
            self.on_finish(req)
        if self.keep_done:
            self.done.append(req)
