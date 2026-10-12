import torch

import tinyserve


def test_package_imports() -> None:
    assert tinyserve.__name__ == "tinyserve"
    assert torch.tensor([1, 2]).sum().item() == 3
