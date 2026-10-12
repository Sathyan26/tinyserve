import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024  # max positions
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 50256

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_json(cls, path: Path) -> "GPT2Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            n_ctx=raw["n_positions"],
            n_embd=raw["n_embd"],
            n_layer=raw["n_layer"],
            n_head=raw["n_head"],
            layer_norm_eps=raw["layer_norm_epsilon"],
            eos_token_id=raw["eos_token_id"],
        )
