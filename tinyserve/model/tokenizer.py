from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer


class Tokenizer:
    """Thin wrapper over the `tokenizers` library (allowed at the edges)."""

    def __init__(self, folder: Path) -> None:
        self._tok = _HFTokenizer.from_file(str(folder / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
