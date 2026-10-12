"""Download and open published checkpoints. Only file I/O lives here; no model code."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

GPT2_REPO = "openai-community/gpt2"
FILES = ("model.safetensors", "config.json", "tokenizer.json")


def download(repo_id: str = GPT2_REPO) -> Path:
    """Fetch weights, config and tokenizer into the HF cache; return the folder holding them."""
    paths = [Path(hf_hub_download(repo_id, name)) for name in FILES]
    return paths[0].parent


def load_state_dict(folder: Path) -> dict[str, Tensor]:
    """Raw tensors from model.safetensors, keyed exactly as the checkpoint names them."""
    return load_file(folder / "model.safetensors")


if __name__ == "__main__":
    folder = download()
    sd = load_state_dict(folder)
    print(f"downloaded to {folder}")
    print(f"{len(sd)} tensors, {sum(t.numel() for t in sd.values()) / 1e6:.1f}M values")
    for name in list(sd)[:16]:
        print(f"  {name:28s} {tuple(sd[name].shape)}")
