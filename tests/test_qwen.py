"""The modern model passes the same equivalence ladder as GPT-2."""

import pytest
import torch

from tinyserve.engine.batch import generate_batch
from tinyserve.engine.generate import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Request, Scheduler
from tinyserve.model.registry import chatml, load
from tinyserve.server.schemas import ChatMessage

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Hi",
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "Explain grouped-query attention in one sentence.",
    "In 1905, Albert Einstein published four papers that changed physics. The first, on the "
    "photoelectric effect, proposed that light is made of discrete quanta.",
]
# Measured max |diff| is exactly 0 on all prompts (same SDPA kernel, same op order for
# RMSNorm and RoPE); the GPT-2 tolerance is kept so a harmless reordering doesn't fail CI.
ATOL = 1e-4

pytestmark = [pytest.mark.slow, pytest.mark.qwen]


@pytest.fixture(scope="module")
def qwen():
    model, tok, chat = load("qwen2.5-0.5b")
    return model, tok, chat


@pytest.fixture(scope="module")
def ref():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_reference(qwen, ref, prompt):
    model, tok, _ = qwen
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        mine, theirs = model(ids), ref(ids).logits
    diff = (mine - theirs).abs().max().item()
    assert diff < ATOL, f"max |diff| = {diff:.2e}"


@pytest.mark.parametrize("penalty", [1.0, 1.1])
def test_greedy_matches_reference(qwen, ref, penalty):
    # Pass repetition_penalty explicitly: HF generate() otherwise applies the checkpoint's
    # generation_config.json (1.1 for Qwen2.5-Instruct) even with do_sample=False.
    model, tok, chat = qwen
    ids = tok.encode(chat.render([ChatMessage(role="user", content="Name three colors.")]))
    p = SamplingParams(max_tokens=24, temperature=0, repetition_penalty=penalty)
    mine = generate(model, ids, p).token_ids
    theirs = ref.generate(
        torch.tensor([ids]), max_new_tokens=24, do_sample=False, repetition_penalty=penalty
    )[0, len(ids) :]
    theirs = [t for t in theirs.tolist() if t not in model.cfg.eos_token_ids]
    assert mine == theirs[: len(mine)]


def test_chat_template_matches_published_one():
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(REPO)
    for msgs in (
        [{"role": "user", "content": "Hello!"}],
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "and 3+3?"},
        ],
    ):
        expected = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert chatml([ChatMessage(**m) for m in msgs]) == expected


def test_special_tokens_are_single_ids(qwen):
    _, tok, _ = qwen
    assert tok.encode("<|im_start|>") == [151644]
    assert tok.encode("<|im_end|>") == [151645]


def test_kv_cache_equals_uncached(qwen):
    model, tok, _ = qwen
    ids = tok.encode(PROMPTS[3])
    p = SamplingParams(max_tokens=20, temperature=0)
    assert (
        generate(model, ids, p, use_cache=True).token_ids
        == generate(model, ids, p, use_cache=False).token_ids
    )


def test_padded_batch_matches_alone(qwen):
    """Left padding + RoPE positions from the mask: the GPT-2 bug class, now with rotations."""
    model, tok, _ = qwen
    prompts = [tok.encode(p) for p in PROMPTS]
    p = SamplingParams(max_tokens=12, temperature=0)
    batched, _ = generate_batch(model, prompts, p)
    for ids, res in zip(prompts, batched, strict=True):
        assert res.token_ids == generate(model, ids, p).token_ids


def test_scheduler_matches_alone(qwen):
    model, tok, _ = qwen
    prompts = [tok.encode(p) for p in PROMPTS]
    params = [SamplingParams(max_tokens=6 + 3 * i, temperature=0) for i in range(len(prompts))]
    sched = Scheduler(model, max_slots=2, max_len=256)
    reqs = [Request(ids, p) for ids, p in zip(prompts, params, strict=True)]
    for r in reqs:
        sched.add(r)
    sched.run_until_done()
    assert sched.cache.k.shape[2] == 2  # only the 2 KV heads are cached, not 14
    for ids, p, r in zip(prompts, params, reqs, strict=True):
        assert r.out == generate(model, ids, p).token_ids
