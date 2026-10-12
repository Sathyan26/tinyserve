"""OpenAI chat-completions wire format (the subset tinyserve supports)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False  # OpenAI: send a final chunk carrying token usage


class ChatCompletionRequest(BaseModel):
    # unknown fields (logprobs=False, user, ...) are ignored, like OpenAI
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt2"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)  # newer name, same meaning
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = Field(default=1, ge=1)
    # tinyserve extensions (OpenAI clients pass them via extra_body)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0)
    ignore_eos: bool = False  # benchmarks: no EOS, no default stop string; exact max_tokens


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
