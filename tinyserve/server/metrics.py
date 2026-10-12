"""Prometheus metrics: what an operator watches, and the raw material for benchmarks.

Gauges are read from the engine at scrape time (nothing to keep in sync). Histograms and
counters are fed by scheduler hooks on the engine thread; prometheus_client is thread-safe.
"""

from collections.abc import Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from tinyserve.engine.engine import Engine
from tinyserve.engine.scheduler import Request


class _EngineGauges(Collector):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def collect(self) -> Iterator[GaugeMetricFamily]:
        s = self.engine.stats()
        for name, doc, value in [
            ("tinyserve_queue_depth", "Submitted, not yet taken by the scheduler", s.queue_depth),
            ("tinyserve_waiting", "Taken by the scheduler, waiting for a slot", s.waiting),
            ("tinyserve_batch_size", "Requests in the current decode batch", s.running),
            ("tinyserve_free_slots", "Free KV cache slots", s.free_slots),
        ]:
            yield GaugeMetricFamily(name, doc, value=value)


class Metrics:
    def __init__(self, engine: Engine) -> None:
        # a private registry: several apps in one process (tests) never collide
        self.registry = CollectorRegistry()
        r = self.registry
        self.ttft = Histogram(
            "tinyserve_ttft_seconds",
            "Time to first token, from request arrival (includes queueing)",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=r,
        )
        self.tpot = Histogram(
            "tinyserve_tpot_seconds",
            "Mean time per output token after the first, per request",
            buckets=(0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1),
            registry=r,
        )
        self.e2e = Histogram(
            "tinyserve_e2e_latency_seconds",
            "Arrival to last token",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
            registry=r,
        )
        self.step_batch = Histogram(
            "tinyserve_decode_batch_size",
            "Sequences per decode step",
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
            registry=r,
        )
        self.tokens = Counter(
            "tinyserve_generated_tokens", "Output tokens generated", registry=r
        )
        self.requests = Counter(
            "tinyserve_requests", "Finished requests", ["finish_reason"], registry=r
        )
        self.rejected = Counter(
            "tinyserve_rejected_requests", "Requests refused at submit", ["reason"], registry=r
        )
        r.register(_EngineGauges(engine))
        engine.scheduler.on_finish = self.observe_finish
        engine.scheduler.on_step = self.step_batch.observe

    def observe_finish(self, req: Request) -> None:
        self.requests.labels(req.finish_reason or "error").inc()
        n = len(req.out)
        self.tokens.inc(n)
        if req.first_token_time is not None:
            self.ttft.observe(req.first_token_time - req.arrival_time)
        if req.finish_time is not None:
            self.e2e.observe(req.finish_time - req.arrival_time)
            if req.first_token_time is not None and n > 1:
                self.tpot.observe((req.finish_time - req.first_token_time) / (n - 1))
