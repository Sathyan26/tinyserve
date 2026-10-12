import pytest
import torch

from tests.conftest import PROMPTS


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_reference(model, ref_model, tokenizer, prompt):
    ids = torch.tensor([tokenizer.encode(prompt)])
    with torch.no_grad():
        mine = model(ids)
        ref = ref_model(ids).logits
    max_diff = (mine - ref).abs().max().item()
    assert torch.allclose(mine, ref, atol=1e-4), f"max |diff| = {max_diff:.2e}"
