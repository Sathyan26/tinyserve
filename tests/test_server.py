import json
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from tinyserve.engine.engine import Engine
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request
from tinyserve.server.app import GPT2_STOP, create_app, gpt2_chat_prompt
from tinyserve.server.schemas import ChatMessage

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
QUESTION = [{"role": "user", "content": "What is the capital of France?"}]


@pytest.fixture(scope="module")
def server(model, tokenizer):
    engine = Engine(model, max_slots=4, max_len=256, max_queue=8, decode=tokenizer.decode)
    app = create_app(engine, tokenizer)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    srv.install_signal_handlers = lambda: None  # only the main thread may install them
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield engine
    srv.should_exit = True
    t.join(10)


def expected_text(model, tokenizer, max_tokens: int) -> str:
    msgs = [ChatMessage(**m) for m in QUESTION]
    ids = tokenizer.encode(gpt2_chat_prompt(msgs))
    p = SamplingParams(max_tokens=max_tokens, temperature=0, stop=(GPT2_STOP,))
    return generate(model, ids, p, tokenizer.decode).text


def sse_events(lines):
    for line in lines:
        if line.startswith("data: "):
            yield line[len("data: ") :]


@pytest.mark.slow
def test_non_stream_matches_engine(server, model, tokenizer):
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": QUESTION, "max_tokens": 20, "temperature": 0},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == expected_text(model, tokenizer, 20)
    assert body["usage"]["completion_tokens"] <= 20


@pytest.mark.slow
def test_stream_equals_non_stream(server):
    payload = {"messages": QUESTION, "max_tokens": 20, "temperature": 0}
    full = httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=60).json()
    with httpx.stream(
        "POST", f"{BASE}/v1/chat/completions", json={**payload, "stream": True}, timeout=60
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = list(sse_events(r.iter_lines()))
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == full["choices"][0]["message"]["content"]


@pytest.mark.slow
def test_official_openai_client_streams(server):
    """Acceptance test: the unmodified OpenAI Python client works against tinyserve."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{BASE}/v1", api_key="unused")
    stream = client.chat.completions.create(
        model="gpt2", messages=QUESTION, max_tokens=20, stream=True, temperature=0
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream)
    assert len(text) > 0

    resp = client.chat.completions.create(
        model="gpt2", messages=QUESTION, max_tokens=20, temperature=0
    )
    assert resp.choices[0].message.content == text
    assert [m.id for m in client.models.list()] == ["gpt2"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "patch",
    [{"n": 2}, {"temperature": 5}, {"messages": []}, {"max_tokens": 10_000}],
    ids=["n", "temperature", "empty", "too-long"],
)
def test_bad_requests_get_openai_style_400(server, patch):
    r = httpx.post(f"{BASE}/v1/chat/completions", json={"messages": QUESTION, **patch}, timeout=30)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.slow
def test_client_disconnect_cancels_generation(server):
    engine = server
    with httpx.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json={"messages": QUESTION, "max_tokens": 200, "stream": True},
        timeout=60,
    ) as r:
        content_chunks = 0
        for e in sse_events(r.iter_lines()):
            if '"content"' in e and '"content":""' not in e:
                content_chunks += 1
            if content_chunks == 3:
                break  # leaving the block closes the connection mid-stream
    deadline = time.monotonic() + 5  # 200 tokens would take ~8 s; cancel must be quicker
    while engine.stats().running or engine.stats().waiting:
        assert time.monotonic() < deadline, "request kept running after disconnect"
        time.sleep(0.05)


def metric(text: str, name: str) -> float:
    """Value of one sample line in Prometheus text format (0 if absent)."""
    for line in text.splitlines():
        sample, _, value = line.rpartition(" ")
        if sample == name:
            return float(value)
    return 0.0


@pytest.mark.slow
def test_metrics_endpoint(server):
    before = httpx.get(f"{BASE}/metrics").text
    for _ in range(2):
        httpx.post(
            f"{BASE}/v1/chat/completions",
            json={"messages": QUESTION, "max_tokens": 8, "temperature": 0},
            timeout=60,
        )
    httpx.post(f"{BASE}/v1/chat/completions", json={"messages": QUESTION, "n": 2})  # a 400
    r = httpx.get(f"{BASE}/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    after = r.text

    def grew(name: str) -> float:
        return metric(after, name) - metric(before, name)

    assert grew("tinyserve_ttft_seconds_count") == 2
    assert grew("tinyserve_e2e_latency_seconds_count") == 2
    assert grew("tinyserve_tpot_seconds_count") == 2
    assert grew("tinyserve_generated_tokens_total") >= 2
    assert grew('tinyserve_rejected_requests_total{reason="invalid"}') == 1
    assert grew("tinyserve_decode_batch_size_count") >= 7  # >= 7 decode steps per request
    for gauge in ("tinyserve_queue_depth", "tinyserve_batch_size", "tinyserve_free_slots"):
        assert gauge in after
    assert metric(after, "tinyserve_free_slots") == 4  # idle again


def test_queue_full_returns_429(model, tokenizer):
    # never started, so the queue stays full
    engine = Engine(model, max_slots=1, max_queue=1, decode=tokenizer.decode)
    engine.submit(Request([1, 2], SamplingParams(max_tokens=2)))
    client = TestClient(create_app(engine, tokenizer))  # no `with`: lifespan (start) skipped
    r = client.post("/v1/chat/completions", json={"messages": QUESTION})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "1"
    assert r.json()["error"]["type"] == "rate_limit_error"
