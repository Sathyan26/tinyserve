"""FastAPI app: OpenAI-compatible /v1/chat/completions in front of the engine thread.

uv run tinyserve            # serves http://127.0.0.1:8000
"""

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tinyserve.engine.engine import Engine, QueueFull
from tinyserve.engine.generate import find_stop
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request as EngineRequest
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.server.detok import TextStreamer
from tinyserve.server.metrics import Metrics
from tinyserve.server.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    ResponseMessage,
    Usage,
)

DEFAULT_MAX_TOKENS = 128
GPT2_STOP = "\nUser:"


def gpt2_chat_prompt(messages: list[ChatMessage]) -> str:
    """GPT-2 has no chat format, so tinyserve uses a plain transcript.

    "User: hi\\nAssistant: hello\\nUser: ...\\nAssistant:" and stops at "\\nUser:".
    """
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = [f"{names[m.role]}: {m.content}" for m in messages]
    return "\n".join(lines) + "\nAssistant:"


def openai_error(status: int, message: str, kind: str, headers: dict | None = None):
    body = {"error": {"message": message, "type": kind, "param": None, "code": None}}
    return JSONResponse(status_code=status, content=body, headers=headers)


def create_app(engine: Engine, tokenizer: Tokenizer, model_name: str = "gpt2") -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="tinyserve", lifespan=lifespan)
    metrics = Metrics(engine)

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    def reject(status: int, message: str) -> JSONResponse:
        """Every refused request goes through here, so the metric cannot miss one."""
        if status == 429:
            metrics.rejected.labels("queue_full").inc()
            return openai_error(429, message, "rate_limit_error", {"Retry-After": "1"})
        metrics.rejected.labels("invalid").inc()
        return openai_error(status, message, "invalid_request_error")

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
        return reject(400, msg)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        return engine.stats().__dict__

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "tinyserve"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        if body.n != 1:
            return reject(400, "n > 1 is not supported")
        prompt_ids = tokenizer.encode(gpt2_chat_prompt(body.messages))
        stops = tuple([body.stop] if isinstance(body.stop, str) else body.stop or [])
        if not body.ignore_eos:
            stops += (GPT2_STOP,)
        room = engine.scheduler.max_len - len(prompt_ids)
        max_tokens = body.max_completion_tokens or body.max_tokens or min(DEFAULT_MAX_TOKENS, room)
        try:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                repetition_penalty=body.repetition_penalty,
                stop=stops,
                seed=body.seed,
                ignore_eos=body.ignore_eos,
            )
            events, req = submit(engine, prompt_ids, params)
        except ValueError as e:
            return reject(400, str(e))
        except QueueFull as e:
            return reject(429, str(e))

        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        if body.stream:
            streamer = TextStreamer(tokenizer.decode, stops)
            usage = body.stream_options is not None and body.stream_options.include_usage
            return StreamingResponse(
                sse(events, req, streamer, cid, created, model_name, usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        finish = None
        try:
            async for _, event_finish in events:
                finish = event_finish  # None until the last event
        finally:
            req.cancelled.set()  # no-op if finished; stops work if this handler was cancelled
        if finish in ("timeout", "error", "cancelled"):
            status = 504 if finish == "timeout" else 500
            return openai_error(status, f"generation ended: {finish}", "server_error")
        text = tokenizer.decode(req.out)  # final once the finish event has arrived
        return ChatCompletion(
            id=cid,
            created=created,
            model=model_name,
            choices=[
                Choice(
                    message=ResponseMessage(content=trim(text, stops)),
                    finish_reason=finish,
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(req.out),
                total_tokens=len(prompt_ids) + len(req.out),
            ),
        )

    return app


def trim(text: str, stops: tuple[str, ...]) -> str:
    cut = find_stop(text, stops)
    return text if cut is None else text[:cut]


Event = tuple[int | None, str | None]


def submit(
    engine: Engine, prompt_ids: list[int], params: SamplingParams
) -> tuple[AsyncIterator[Event], EngineRequest]:
    """Hand a request to the engine thread; return an async iterator of its events.

    Submitting happens *before* any response is sent, so a full queue or bad request
    still becomes a proper 429/400 instead of a broken stream.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Event] = asyncio.Queue()

    def on_token(tok: int | None, finish: str | None) -> None:
        # runs on the ENGINE thread: never touch asyncio objects directly from here
        try:
            loop.call_soon_threadsafe(q.put_nowait, (tok, finish))
        except RuntimeError:  # event loop already closed (server shutting down)
            req.cancelled.set()

    req = EngineRequest(prompt_ids, params, on_token=on_token)
    engine.submit(req)

    async def events() -> AsyncIterator[Event]:
        while True:
            tok, finish = await q.get()
            yield tok, finish
            if finish is not None:
                return

    return events(), req


async def sse(
    events: AsyncIterator[Event],
    req: EngineRequest,
    streamer: TextStreamer,
    cid: str,
    created: int,
    model: str,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    def chunk(delta: Delta, finish: str | None = None) -> str:
        c = ChatCompletionChunk(
            id=cid,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=delta, finish_reason=finish)],
        )
        return f"data: {c.model_dump_json(exclude_none=True)}\n\n"

    try:
        yield chunk(Delta(role="assistant", content=""))
        async for tok, finish in events:
            if tok is not None:
                if text := streamer.push(tok):
                    yield chunk(Delta(content=text))
                continue
            if finish in ("timeout", "error", "cancelled"):
                err = {"error": {"message": f"generation ended: {finish}", "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"
            else:
                if text := streamer.finish():
                    yield chunk(Delta(content=text))
                yield chunk(Delta(), finish)
        if include_usage:  # OpenAI's shape: one extra chunk, empty choices, usage set
            n_prompt, n_out = len(req.prompt_ids), len(req.out)
            last = ChatCompletionChunk(
                id=cid,
                created=created,
                model=model,
                choices=[],
                usage=Usage(
                    prompt_tokens=n_prompt, completion_tokens=n_out, total_tokens=n_prompt + n_out
                ),
            )
            yield f"data: {last.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # client went away (Starlette cancels this generator) or we finished: either way
        # the engine drops the request at its next step and frees the slot
        req.cancelled.set()


def build_from_env() -> FastAPI:
    from tinyserve.model.gpt2 import GPT2
    from tinyserve.model.weights import download

    folder = download()
    tokenizer = Tokenizer(folder)
    engine = Engine(
        GPT2.from_pretrained(folder),
        max_slots=int(os.environ.get("TINYSERVE_MAX_SLOTS", 8)),
        max_len=int(os.environ.get("TINYSERVE_MAX_LEN", 1024)),
        max_queue=int(os.environ.get("TINYSERVE_MAX_QUEUE", 64)),
        request_timeout_s=float(os.environ.get("TINYSERVE_TIMEOUT_S", 120)),
        decode=tokenizer.decode,
        mode=os.environ.get("TINYSERVE_SCHEDULER_MODE", "continuous"),
    )
    return create_app(engine, tokenizer)


def main() -> None:
    import uvicorn

    uvicorn.run(
        build_from_env(),
        host=os.environ.get("TINYSERVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("TINYSERVE_PORT", 8000)),
    )
