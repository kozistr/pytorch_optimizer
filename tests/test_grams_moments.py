"""GRAMS tracks a real first moment and a debiased Adam denominator."""
import torch

from pytorch_optimizer.optimizer.grams import Grams


def test_first_moment_keeps_most_of_its_history():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Grams([param], lr=0.1)
    param.grad = torch.tensor([0.4])
    opt.step()
    # (1 - 0.9) times the first gradient, not 0.9 times it.
    assert torch.allclose(opt.state[param]['exp_avg'], torch.tensor([0.04]))


def test_first_step_matches_hand_computation():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = Grams([param], lr=0.1, betas=(0.9, 0.999), eps=1e-6)
    param.grad = torch.tensor([0.4, -0.2])
    opt.step()
    # m = 0.1 g, v = 0.001 g^2. Debiased: m / 0.1 = g,
    # sqrt(v) / sqrt(0.001) = |g|. |g / (|g| + eps)| is ~1 with the
    # sign of g, so the parameter moves ~lr along -sign(g).
    assert torch.allclose(param.detach(), torch.tensor([0.9, 2.1]), atol=1e-4)
