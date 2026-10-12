"""Perplexity on WikiText-2 (test split): the quality axis of the quantization trade-off.

Sliding window (as in the Hugging Face perplexity guide): each window sees up to `window`
tokens of context, but only the tokens not scored by the previous window count toward the
loss, so every token is scored exactly once and most have long context.
"""

from collections.abc import Callable

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def load_wikitext2_test() -> str:
    path = hf_hub_download(WIKITEXT_REPO, WIKITEXT_FILE, repo_type="dataset")
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())


@torch.no_grad()
def perplexity(
    logits_fn: Callable[[Tensor], Tensor],
    ids: list[int],
    window: int = 1024,
    stride: int = 512,
) -> float:
    """exp(mean negative log-likelihood) over `ids`. logits_fn: [1, T] -> [1, T, vocab]."""
    tokens = torch.tensor(ids)
    nll_sum, counted, prev_end = 0.0, 0, 0
    for begin in range(0, len(tokens), stride):
        end = min(begin + window, len(tokens))
        new = end - prev_end  # tokens this window scores for the first time
        chunk = tokens[begin:end][None]
        logits = logits_fn(chunk)[0].float()
        # position t predicts token t+1; score only the last `new` targets
        targets = chunk[0, 1:]
        logp = logits[:-1]
        n = min(new, len(targets))
        nll_sum += F.cross_entropy(logp[-n:], targets[-n:], reduction="sum").item()
        counted += n
        prev_end = end
        if end == len(tokens):
            break
    return float(torch.exp(torch.tensor(nll_sum / counted)))
