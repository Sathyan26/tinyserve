"""GPT-2 written from scratch. Only torch is used; weights come from the published checkpoint."""

from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyserve.model.config import GPT2Config
from tinyserve.model.weights import load_state_dict


class Cache(Protocol):
    """Anything that stores past keys/values. Implemented in engine/kv_cache.py."""

    @property
    def length(self) -> int:
        """Past positions in the longest row (keys already stored before this forward)."""
        ...

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append this step's k, v [B, H, T, D] for `layer`; return the full history."""
        ...

    def advance(self, T: int) -> None:
        """Called once after the last layer: the T new positions are now stored."""
        ...


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v packed in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: Tensor, mask: Tensor, cache: Cache | None, layer: int) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, T, C] -> [B, H, T, D]: each head attends independently
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k, v = cache.update(layer, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


def causal_mask(T: int, S: int, device: torch.device) -> Tensor:
    """Bool [T, S]: query i sits at absolute position past+i and may see keys 0..past+i.

    Built by hand because SDPA's is_causal aligns top-left, which is wrong once past > 0.
    """
    past = S - T
    q_pos = torch.arange(past, S, device=device)[:, None]
    k_pos = torch.arange(S, device=device)[None, :]
    return k_pos <= q_pos


def attention_mask(T: int, key_mask: Tensor | None, S: int, device: torch.device) -> Tensor:
    """Combine causality with padding. key_mask [B, S] is True for real tokens.

    Returns bool [B, 1, T, S] (or [T, S] without padding). A padding *query* whose row
    would be all-False gets to see itself. A hand-written softmax over an empty row gives
    NaN, which spreads to the whole batch. torch>=2.5 SDPA returns zeros there instead
    (checked in LOG.md), so this guard only matters for custom attention, but it is cheap.
    """
    causal = causal_mask(T, S, device)
    if key_mask is None:
        return causal
    mask = causal[None] & key_mask[:, None, :]  # [B, T, S]
    self_key = torch.zeros(T, S, dtype=torch.bool, device=device)
    self_key[torch.arange(T), torch.arange(S - T, S)] = True
    mask = mask | (self_key[None] & ~mask.any(-1, keepdim=True))  # only rows that are empty
    return mask[:, None]


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        # GPT-2 uses the tanh approximation of GELU; exact GELU drifts 7.5e-2 from reference
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, mask: Tensor, cache: Cache | None, layer: int) -> Tensor:
        x = x + self.attn(self.ln_1(x), mask, cache, layer)  # pre-norm residual
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)  # learned position embeddings
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        cache: Cache | None = None,
        positions: Tensor | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """input_ids [B, T] -> logits [B, T, vocab].

        positions [B, T]: absolute position of each token. Defaults to past..past+T, which is
        only right when no row is padded; batched callers pass their own.
        key_mask [B, past+T]: True for real tokens, False for padding (None = no padding).
        """
        T = input_ids.size(1)
        past = cache.length if cache is not None else 0
        if positions is None:
            positions = torch.arange(past, past + T, device=input_ids.device)[None]
        mask = attention_mask(T, key_mask, past + T, input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)
        for i, block in enumerate(self.h):
            x = block(x, mask, cache, i)
        if cache is not None:
            cache.advance(T)  # once per forward pass, never per layer
        # output head is tied to the token embedding: no separate lm_head weight exists
        return self.ln_f(x) @ self.wte.weight.T

    @property
    def dtype(self) -> torch.dtype:
        """Activation dtype; KV caches are allocated in it (int8 models still compute in fp32)."""
        return self.wte.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.wte.weight.device

    @classmethod
    def from_pretrained(cls, folder: Path) -> "GPT2":
        model = cls(GPT2Config.from_json(folder / "config.json"))
        load_gpt2_weights(model, load_state_dict(folder))
        return model.eval()


# Checkpoint stores these as Conv1D with shape [in, out]; nn.Linear wants [out, in].
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2_weights(model: GPT2, sd: dict[str, Tensor]) -> None:
    """Copy checkpoint tensors into `model`, failing loudly on anything missing or unused."""
    params = dict(model.named_parameters())
    filled: set[str] = set()
    with torch.no_grad():
        for name, tensor in sd.items():
            if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
                continue  # cached causal masks, not weights
            if name not in params:
                raise KeyError(f"checkpoint key {name!r} has no matching parameter")
            if name.endswith(_TRANSPOSED):
                tensor = tensor.T
            if params[name].shape != tensor.shape:
                raise ValueError(f"{name}: model {params[name].shape} vs checkpoint {tensor.shape}")
            params[name].copy_(tensor)
            filled.add(name)
    missing = set(params) - filled
    if missing:
        raise KeyError(f"parameters never loaded: {sorted(missing)}")
