import pytest

from tinyserve.server.app import gpt2_chat_prompt
from tinyserve.server.detok import REPLACEMENT, TextStreamer, stop_prefix_len
from tinyserve.server.schemas import ChatMessage


def stream_all(tokenizer, text, stops=()):
    s = TextStreamer(tokenizer.decode, stops)
    deltas = [s.push(t) for t in tokenizer.encode(text)]
    deltas.append(s.finish())
    return deltas


def test_multibyte_characters_never_split(tokenizer):
    text = "I love 🍕 and 東京!"
    deltas = stream_all(tokenizer, text)
    assert "".join(deltas) == text
    assert not any(REPLACEMENT in d for d in deltas)
    # the pizza emoji really does span several tokens, so the test is meaningful
    ids = tokenizer.encode(" 🍕")
    assert len(ids) > 1 and REPLACEMENT in tokenizer.decode(ids[:1])


def test_stop_string_is_never_half_streamed(tokenizer):
    deltas = stream_all(tokenizer, "Hi there\nUser: ok", stops=("\nUser:",))
    assert "".join(deltas) == "Hi there"
    assert not any("\n" in d for d in deltas)  # "\n" was held back, then dropped


def test_held_back_text_is_released_when_no_stop_follows(tokenizer):
    deltas = stream_all(tokenizer, "a\nUsual day", stops=("\nUser:",))
    assert "".join(deltas) == "a\nUsual day"


def test_stop_prefix_len():
    assert stop_prefix_len("abc\nUs", ("\nUser:",)) == 3
    assert stop_prefix_len("abc", ("\nUser:",)) == 0
    assert stop_prefix_len("x\n", ("\nUser:", "\n\n")) == 1


def test_chat_template():
    msgs = [ChatMessage(role="system", content="Be brief."), ChatMessage(role="user", content="Hi")]
    assert gpt2_chat_prompt(msgs) == "System: Be brief.\nUser: Hi\nAssistant:"


def test_vllm_gpt2_template_matches_tinyserve():
    """The ceiling benchmark must send vLLM byte-identical prompts."""
    from pathlib import Path

    jinja2 = pytest.importorskip("jinja2")
    src = (Path(__file__).parents[1] / "templates" / "gpt2_chat.jinja").read_text()
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]
    rendered = jinja2.Template(src).render(messages=msgs)
    assert rendered == gpt2_chat_prompt([ChatMessage(**m) for m in msgs])
