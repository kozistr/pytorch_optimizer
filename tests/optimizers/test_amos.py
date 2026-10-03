import torch

from pytorch_optimizer.optimizer import load_optimizer


def test_amos_get_scale():
    opt = load_optimizer('amos')

    assert opt.get_scale(torch.zeros((1,))) == 0.5
    assert opt.get_scale(torch.zeros((1, 4))) == 0.7071067811865476
    assert opt.get_scale(torch.zeros((1, 16, 2, 2))) == 0.25
