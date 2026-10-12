"""Run a tinyserve app inside the current process (benchmarks and tests)."""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI


@contextmanager
def serve_in_thread(app: FastAPI, port: int, host: str = "127.0.0.1") -> Iterator[str]:
    """Start uvicorn on a background thread; yield the base URL; stop it on exit."""
    srv = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    srv.install_signal_handlers = lambda: None  # type: ignore[method-assign]  # main thread only
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 60
    while not srv.started:
        if not t.is_alive() or time.monotonic() > deadline:
            raise RuntimeError(f"server on port {port} did not start")
        time.sleep(0.05)
    try:
        yield f"http://{host}:{port}"
    finally:
        srv.should_exit = True
        t.join(30)
