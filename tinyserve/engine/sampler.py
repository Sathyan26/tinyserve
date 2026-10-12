"""Turn one sequence's next-token logits into a token id.

Processing order matches Hugging Face (penalty -> temperature -> top-k -> top-p) so greedy
output can be compared token for token with the reference.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0  # 0 means greedy
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    stop: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def make_generator(params: SamplingParams) -> torch.Generator:
    """Each request owns its RNG, so its samples never depend on what else is in the batch."""
    g = torch.Generator()
    g.manual_seed(params.seed if params.seed is not None else torch.seed())
    return g


def apply_repetition_penalty(logits: Tensor, seen: Tensor, penalty: float) -> Tensor:
    """CTRL-style penalty on every token already in the sequence (prompt included)."""
    if penalty == 1.0 or seen.numel() == 0:
        return logits
    scores = logits.gather(0, seen)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(0, seen, scores)


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    if k <= 0 or k >= logits.size(0):
        return logits
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens whose probability mass reaches p (always >= 1 token)."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    drop = cum - sorted_logits.softmax(-1) >= p  # mass *before* this token already reached p
    drop[0] = False
    return logits.masked_fill(drop.scatter(0, order, drop), float("-inf"))


def sample(logits: Tensor, seen: Tensor, params: SamplingParams, g: torch.Generator) -> int:
    """logits [vocab] for one sequence, seen [n] token ids so far -> next token id."""
    logits = apply_repetition_penalty(logits.float(), seen, params.repetition_penalty)
    if params.greedy:
        return int(logits.argmax())
    logits = logits / params.temperature
    logits = apply_top_k(logits, params.top_k)
    logits = apply_top_p(logits, params.top_p)
    return int(torch.multinomial(logits.softmax(-1), 1, generator=g))
