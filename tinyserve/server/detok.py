"""Incremental detokenization for streaming.

Two traps handled here:
1. One character can span several tokens (emoji, CJK). Decoding a half character yields
   U+FFFD, so nothing is sent while the text ends in one.
2. A stop string arrives over several tokens. Text that could still turn into a stop
   string is held back, so "\\nUser:" is never half-streamed and then retracted.
"""

from collections.abc import Callable

from tinyserve.engine.generate import find_stop

REPLACEMENT = "�"


def stop_prefix_len(text: str, stops: tuple[str, ...]) -> int:
    """Length of the longest tail of `text` that is a proper prefix of some stop string."""
    best = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:k]):
                best = k
                break
    return best


class TextStreamer:
    def __init__(self, decode: Callable[[list[int]], str], stops: tuple[str, ...] = ()) -> None:
        self.decode = decode
        self.stops = stops
        self.ids: list[int] = []
        self.sent = 0  # characters already sent to the client

    def push(self, token_id: int) -> str:
        """Add one token; return the text that is now safe to send (may be empty)."""
        self.ids.append(token_id)
        text = self.decode(self.ids)
        if text.endswith(REPLACEMENT):
            return ""
        end = len(text) - stop_prefix_len(text, self.stops)
        if (cut := find_stop(text, self.stops)) is not None:
            end = min(end, cut)
        return self._emit(text, end)

    def finish(self) -> str:
        """Generation ended: flush whatever is left, minus any stop string."""
        text = self.decode(self.ids).rstrip(REPLACEMENT)  # a char cut off by max_tokens
        cut = find_stop(text, self.stops)
        return self._emit(text, len(text) if cut is None else cut)

    @property
    def text(self) -> str:
        return self.decode(self.ids)

    def _emit(self, text: str, end: int) -> str:
        if end <= self.sent:
            return ""
        delta = text[self.sent : end]
        self.sent = end
        return delta
