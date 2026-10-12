"""The scheduler running in its own thread, fed by a bounded request queue.

Web handlers call submit() from the server's threads; only the engine thread ever touches
the model, the KV cache or the scheduler's lists, so none of them need locks.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from tinyserve.engine.scheduler import Request, Scheduler
from tinyserve.model.gpt2 import GPT2

log = logging.getLogger(__name__)


class QueueFull(Exception):
    """The request queue is at capacity; the server answers 429."""


@dataclass(frozen=True)
class EngineStats:
    queue_depth: int  # submitted, not yet taken by the scheduler
    waiting: int  # taken, waiting for a free slot
    running: int  # current decode batch size
    free_slots: int
    steps: int


class Engine:
    def __init__(
        self,
        model: GPT2,
        max_slots: int = 8,
        max_len: int | None = None,
        max_queue: int = 64,
        request_timeout_s: float | None = 120.0,
        decode: Callable[[list[int]], str] | None = None,
        idle_poll_s: float = 0.05,
        mode: str = "continuous",
        batch_wait_s: float = 0.2,
    ) -> None:
        self.scheduler = Scheduler(
            model,
            max_slots,
            max_len,
            decode,
            keep_done=False,
            mode=mode,
            batch_wait_s=batch_wait_s,
        )
        self.queue: queue.Queue[Request] = queue.Queue(maxsize=max_queue)
        self.request_timeout_s = request_timeout_s
        self.idle_poll_s = idle_poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0

    # ---- called from other threads ----

    def submit(self, req: Request) -> None:
        """Validate and enqueue. Raises ValueError (bad request) or QueueFull (overloaded)."""
        self.scheduler.validate(req)  # reads only immutable config: safe off-thread
        if req.deadline is None and self.request_timeout_s is not None:
            req.deadline = time.perf_counter() + self.request_timeout_s
        try:
            self.queue.put_nowait(req)
        except queue.Full:
            raise QueueFull(f"queue is full ({self.queue.maxsize} requests)") from None

    def start(self) -> "Engine":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="tinyserve-engine", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def stats(self) -> EngineStats:
        s = self.scheduler  # racy snapshot of ints and list lengths; fine for metrics
        return EngineStats(
            queue_depth=self.queue.qsize(),
            waiting=len(s.waiting),
            running=len(s.running),
            free_slots=s.cache.num_free,
            steps=s.steps,
        )

    # ---- engine thread only ----

    def _loop(self) -> None:
        s = self.scheduler
        while not self._stop.is_set():
            if not s.has_work:
                try:  # idle: block briefly instead of spinning a CPU core
                    s.waiting.append(self.queue.get(timeout=self.idle_poll_s))
                except queue.Empty:
                    continue
            self._pull()
            try:
                if not s.step():
                    # nothing decoded (e.g. static mode holding a partial batch): wait for
                    # an arrival instead of spinning a core and starving the server thread
                    self._wait_for_arrival(0.005)
            except Exception:
                # one bad step must not kill the thread and hang every client
                self.errors += 1
                log.exception("engine step failed; failing in-flight requests")
                s.fail_all("error")
        s.fail_all("cancelled")  # shutting down

    def _wait_for_arrival(self, timeout: float) -> None:
        s = self.scheduler
        if len(s.waiting) < s.cache.num_free:
            try:
                s.waiting.append(self.queue.get(timeout=timeout))
            except queue.Empty:
                pass
        else:
            time.sleep(timeout)

    def _pull(self) -> None:
        """Move requests from the queue into the scheduler only as slots allow.

        Draining everything would make the queue bound meaningless: the scheduler's own
        waiting list would grow without limit instead.
        """
        s = self.scheduler
        while len(s.waiting) < s.cache.num_free:
            try:
                s.waiting.append(self.queue.get_nowait())
            except queue.Empty:
                return
