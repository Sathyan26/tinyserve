"""Key/value cache: past keys and values per layer, so each new token costs one step."""

import torch
from torch import Tensor


class KVCache:
    """Contiguous cache for one batch of sequences that all share the same length.

    Allocated once up front: [n_layer, B, n_head, max_len, head_dim] for k and for v.
    """

    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_head: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (n_layer, batch, n_head, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0  # positions filled; advanced by the model after the last layer

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        start, end = self.length, self.length + k.size(2)
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > max_len {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, T: int) -> None:
        self.length += T

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class SlotKVCache:
    """One preallocated pool shared by every running request; each request owns one slot.

    k, v: [n_layer, max_slots, n_head, max_len, head_dim]. Requests join (alloc) and leave
    (free) without moving anyone else's memory. Cost: every slot reserves max_len positions
    whether it uses them or not, which is the waste paged KV memory (vLLM) removes.
    """

    def __init__(
        self,
        n_layer: int,
        max_slots: int,
        n_head: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (n_layer, max_slots, n_head, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.lengths = torch.zeros(max_slots, dtype=torch.long)  # positions filled per slot
        self._free = list(range(max_slots - 1, -1, -1))  # pop() hands out slot 0 first

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("no free KV slot")
        slot = self._free.pop()
        self.lengths[slot] = 0  # a reused slot must not inherit the previous owner's length
        return slot

    def free(self, slot: int) -> None:
        if slot in self._free:
            raise RuntimeError(f"slot {slot} freed twice")
        self._free.append(slot)

    def view(self, slots: Tensor, T: int) -> "SlotView":
        return SlotView(self, slots, T)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class SlotView:
    """The Cache seen by one forward pass over some slots.

    Decode (T == 1): any number of slots, each at its own length.
    Prefill (T > 1): exactly one slot, so no row needs padding.
    """

    def __init__(self, cache: SlotKVCache, slots: Tensor, T: int) -> None:
        if T > 1 and slots.numel() != 1:
            raise ValueError("prefill (T > 1) runs one slot at a time")
        self.cache = cache
        self.slots = slots
        self.T = T
        self.start = cache.lengths[slots].clone()  # [B] where each row's new tokens go
        self.L = int(self.start.max()) + T  # keys to attend over (longest row)
        if self.L > cache.max_len:
            raise RuntimeError(f"KV slot overflow: {self.L} > max_len {cache.max_len}")

    @property
    def length(self) -> int:
        return self.L - self.T

    def positions(self) -> Tensor:
        """[B, T] absolute position of each new token."""
        return self.start[:, None] + torch.arange(self.T)[None]

    def key_mask(self) -> Tensor:
        """[B, L] True where a key belongs to that row (shorter rows ignore the tail)."""
        return torch.arange(self.L)[None] < (self.start + self.T)[:, None]

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        c, s = self.cache, self.slots
        if self.T == 1:
            # advanced indexing: row b writes position start[b] of slot s[b]
            c.k[layer, s, :, self.start] = k[:, :, 0]
            c.v[layer, s, :, self.start] = v[:, :, 0]
        else:
            a = int(self.start[0])
            c.k[layer, s[0], :, a : a + self.T] = k[0]
            c.v[layer, s[0], :, a : a + self.T] = v[0]
        # gather [B, H, L, D]: a copy per layer per step, the price of non-contiguous slots
        return c.k[layer, s, :, : self.L], c.v[layer, s, :, : self.L]

    def advance(self, T: int) -> None:
        self.cache.lengths[self.slots] += T
