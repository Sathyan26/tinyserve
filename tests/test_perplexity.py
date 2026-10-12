import pytest
import torch

from tinyserve.bench.perplexity import load_wikitext2_test, perplexity


def test_perplexity_of_a_perfect_model_is_one():
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def oracle(chunk):  # puts all mass on the true next token
        logits = torch.full((1, chunk.size(1), 16), -1e4)
        for t in range(chunk.size(1) - 1):
            logits[0, t, chunk[0, t + 1]] = 1e4
        return logits

    assert perplexity(oracle, ids, window=4, stride=2) == pytest.approx(1.0)


def test_uniform_model_has_vocab_perplexity():
    ids = list(range(20))
    assert perplexity(lambda c: torch.zeros(1, c.size(1), 50), ids, window=8, stride=3) == (
        pytest.approx(50.0, rel=1e-5)
    )


def test_every_token_scored_once():
    seen: list[int] = []

    def spy(chunk):
        seen.append(chunk.size(1))
        return torch.zeros(1, chunk.size(1), 4)

    ids = list(range(10))
    perplexity(spy, [i % 4 for i in ids], window=4, stride=2)
    assert seen == [4, 4, 4, 4]  # windows at 0, 2, 4, 6 (last one reaches the end)


@pytest.mark.slow
def test_matches_reference_model(model, ref_model, tokenizer):
    """Our model + our perplexity code == the HF model through the same code."""
    ids = tokenizer.encode(load_wikitext2_test())[:3000]
    mine = perplexity(lambda c: model(c), ids)
    ref = perplexity(lambda c: ref_model(c).logits, ids)
    assert mine == pytest.approx(ref, rel=1e-4)
    assert 10 < mine < 60  # GPT-2 small on WikiText-2 is ~25-30
