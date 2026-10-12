import pytest

from tinyserve.model.gpt2 import GPT2
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import download

PROMPTS = [
    "Hello",
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "¿Dónde está la biblioteca? Je ne sais pas, 東京 maybe!",
    "In 1905, Albert Einstein published four papers that changed physics. The first, on the "
    "photoelectric effect, proposed that light is made of discrete quanta. The second explained "
    "Brownian motion, giving strong evidence that atoms exist. The third introduced special",
]


@pytest.fixture(scope="session")
def gpt2_folder():
    return download()


@pytest.fixture(scope="session")
def model(gpt2_folder):
    return GPT2.from_pretrained(gpt2_folder)


@pytest.fixture(scope="session")
def tokenizer(gpt2_folder):
    return Tokenizer(gpt2_folder)


@pytest.fixture(scope="session")
def ref_model():
    # transformers is a dev-only dependency: used here as the reference, never in the engine
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
