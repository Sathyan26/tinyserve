"""Every supported model in one place: how to load it, tokenize for it and format chats."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import download


class Message(Protocol):
    role: str
    content: str


class LanguageModel(Protocol):
    """What the engine needs from a model. GPT2 and Qwen2 both satisfy it."""

    cfg: object  # n_layer, n_kv_head, head_dim, n_ctx, eos_token_ids

    @property
    def dtype(self) -> torch.dtype: ...

    def __call__(
        self,
        input_ids: Tensor,
        cache: object = None,
        positions: Tensor | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor: ...


@dataclass(frozen=True)
class ChatFormat:
    render: Callable[[list[Message]], str]
    stops: tuple[str, ...]  # stop strings added to every chat request


def gpt2_chat(messages: list[Message]) -> str:
    """GPT-2 has no chat format, so tinyserve uses a plain transcript.

    "User: hi\\nAssistant: hello\\nUser: ...\\nAssistant:" and stops at "\\nUser:".
    """
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = [f"{names[m.role]}: {m.content}" for m in messages]
    return "\n".join(lines) + "\nAssistant:"


QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def chatml(messages: list[Message]) -> str:
    """Qwen2.5's ChatML template (the published one adds this system prompt when none is
    given). Generation ends on the <|im_end|> token, so no stop strings are needed."""
    parts = []
    if not messages or messages[0].role != "system":
        parts.append(f"<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n")
    parts += [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    chat: ChatFormat


MODELS = {
    "gpt2": ModelSpec("openai-community/gpt2", ChatFormat(gpt2_chat, ("\nUser:",))),
    "qwen2.5-0.5b": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", ChatFormat(chatml, ())),
}


def load(
    name: str, dtype: torch.dtype = torch.float32
) -> tuple[LanguageModel, Tokenizer, ChatFormat]:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]
    folder = download(spec.repo)
    if name == "gpt2":
        from tinyserve.model.gpt2 import GPT2

        model: LanguageModel = GPT2.from_pretrained(folder).to(dtype)
    else:
        from tinyserve.model.qwen2 import Qwen2

        model = Qwen2.from_pretrained(folder, dtype=dtype)
    return model, Tokenizer(folder), spec.chat
