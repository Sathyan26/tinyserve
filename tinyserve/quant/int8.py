"""Weight-only quantization, written by hand: int8 per-output-channel and fp16/bf16 casts.

Only the Linear layers inside the transformer blocks are quantized (85M of GPT-2's 124M
parameters). The tied token embedding / output head and the position table stay in full
precision: the head decides every token, and the embedding is a lookup, not a matmul.
"""

import copy
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from tinyserve.model.gpt2 import GPT2


class QuantLinear(nn.Module):
    """y = x @ W.T + b with W stored as int8 plus one fp scale per output row.

    Symmetric per-channel: scale_o = max|W_o| / 127, W_int8 = round(W / scale).
    The worst-case error per weight is scale_o / 2.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("w_int8", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).to(linear.weight.dtype))  # [out]
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        # Scale is per OUTPUT channel, so it can be applied after the matmul:
        # x @ (Wq * s).T == (x @ Wq.T) * s. Plain PyTorch has no int8 x float kernel, so the
        # int8 weights are still widened to x.dtype on every call: memory falls, speed doesn't.
        y = (x @ self.w_int8.to(x.dtype).T) * self.scale.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)

    def dequantized(self) -> Tensor:
        return self.w_int8.float() * self.scale.float()[:, None]


def quantize_int8(model: GPT2, layers: str | Iterable[int] = "all", inplace: bool = False) -> GPT2:
    """Replace every nn.Linear in the chosen transformer blocks with QuantLinear.

    `layers` takes block indices, so experiments can quantize one block at a time.
    """
    model = model if inplace else copy.deepcopy(model)
    chosen = range(len(model.h)) if layers == "all" else layers
    for i in chosen:
        block = model.h[i]
        for parent in (block.attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, QuantLinear(child))
    return model


def cast(model: GPT2, dtype: torch.dtype, inplace: bool = False) -> GPT2:
    """fp16 / bf16: every weight in the lower precision (the simplest quantization)."""
    model = model if inplace else copy.deepcopy(model)
    return model.to(dtype)


def model_bytes(model: nn.Module) -> int:
    """Bytes of every parameter and buffer, i.e. what the weights cost in memory."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
