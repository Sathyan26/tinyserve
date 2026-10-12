"""Qwen2 / Qwen2.5 written from scratch: RoPE, RMSNorm, grouped-query attention, SwiGLU.

Same interface as GPT2 (cfg, dtype, forward(input_ids, cache, positions, key_mask)), so the
engine, scheduler, server and benchmarks run it unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyserve.model.gpt2 import Cache, attention_mask
from tinyserve.model.weights import load_state_dict


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    n_embd: int  # hidden size
    n_layer: int
    n_head: int  # query heads
    n_kv_head: int  # key/value heads (GQA: several query heads share one)
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    n_ctx: int
    eos_token_ids: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_folder(cls, folder: Path) -> "Qwen2Config":
        raw = json.loads((folder / "config.json").read_text())
        gen_path = folder / "generation_config.json"
        eos = raw["eos_token_id"]
        if gen_path.exists():  # the chat model stops on <|im_end|> AND <|endoftext|>
            eos = json.loads(gen_path.read_text()).get("eos_token_id", eos)
        if not raw.get("tie_word_embeddings", False):
            raise NotImplementedError("only tied-embedding Qwen2 checkpoints are supported")
        return cls(
            vocab_size=raw["vocab_size"],
            n_embd=raw["hidden_size"],
            n_layer=raw["num_hidden_layers"],
            n_head=raw["num_attention_heads"],
            n_kv_head=raw["num_key_value_heads"],
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            n_ctx=raw["max_position_embeddings"],
            eos_token_ids=tuple(eos) if isinstance(eos, list) else (eos,),
        )


class RMSNorm(nn.Module):
    """x / rms(x) * w. Computed in fp32 like the reference, then cast back."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def rope_cos_sin(positions: Tensor, inv_freq: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """positions [B, T] -> cos, sin [B, 1, T, head_dim], in fp32 then cast (as the reference)."""
    freqs = positions[..., None].float() * inv_freq  # [B, T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # "rotate half" layout, not interleaved pairs
    return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        d = cfg.head_dim
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, d
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * d, bias=True)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * d, bias=True)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * d, bias=True)
        self.o_proj = nn.Linear(cfg.n_head * d, cfg.n_embd, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, cache: Cache | None, layer: int
    ) -> Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, -1).transpose(1, 2)
        # rotate BEFORE caching: the cache stores keys already at their absolute position
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            k, v = cache.update(layer, k, v)  # cache holds only the 2 KV heads, not 14
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, cache, layer):  # type: ignore[no-untyped-def]
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache, layer)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)

    def inv_freq(self, device: torch.device) -> Tensor:
        """RoPE frequencies, always fp32. Kept out of the buffers on purpose: model.to(bf16)
        would round them, shifting rotation angles at large positions."""
        d = self.cfg.head_dim
        exps = torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d
        return 1.0 / (self.cfg.rope_theta**exps)

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def forward(
        self,
        input_ids: Tensor,
        cache: Cache | None = None,
        positions: Tensor | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """input_ids [B, T] -> logits [B, T, vocab]. Same contract as GPT2.forward."""
        B, T = input_ids.shape
        past = cache.length if cache is not None else 0
        if positions is None:
            positions = torch.arange(past, past + T, device=input_ids.device)[None].expand(B, T)
        mask = attention_mask(T, key_mask, past + T, input_ids.device)
        x = self.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(positions, self.inv_freq(x.device), x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, cache, i)
        if cache is not None:
            cache.advance(T)
        return self.norm(x) @ self.embed_tokens.weight.T  # tied output head

    @classmethod
    def from_pretrained(cls, folder: Path, dtype: torch.dtype = torch.float32) -> "Qwen2":
        model = cls(Qwen2Config.from_folder(folder))
        sd = load_state_dict(folder)
        params = dict(model.named_parameters())
        filled = set()
        with torch.no_grad():
            for name, tensor in sd.items():
                ours = name.removeprefix("model.")
                if ours not in params:
                    raise KeyError(f"checkpoint key {name!r} has no matching parameter")
                if params[ours].shape != tensor.shape:
                    raise ValueError(f"{name}: {params[ours].shape} vs {tensor.shape}")
                params[ours].copy_(tensor)  # bf16 checkpoint -> fp32 params (exact widening)
                filled.add(ours)
        missing = set(params) - filled
        if missing:
            raise KeyError(f"parameters never loaded: {sorted(missing)}")
        return model.to(dtype).eval()
