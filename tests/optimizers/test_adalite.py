import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_fixed_decay():
    params = [torch.ones(2, 2, requires_grad=True) for _ in range(2)]
    optimizers = [
        build_optimizer('adalite', [p], lr=0.1, weight_decay=decay, weight_decouple=True, fixed_decay=True)
        for p, decay in zip(params, (0.1, 0.0))
    ]

    for p, optimizer in zip(params, optimizers):
        p.grad = torch.full_like(p, 0.5)
        optimizer.step()

    torch.testing.assert_close(params[0], params[1] - 0.1)


def test_convolution_matches_matrix():
    conv = make_parameter((2, 3, 2))
    matrix = make_parameter((2, 6))

    with torch.no_grad():
        conv.copy_(torch.arange(12).reshape_as(conv) / 10.0)
        matrix.copy_(conv.reshape_as(matrix))

    optimizers = [build_optimizer('adalite', [p]) for p in (conv, matrix)]

    for step in range(2):
        gradient = torch.sin(torch.arange(12) + step)

        for p, optimizer in zip((conv, matrix), optimizers):
            p.grad = gradient.reshape_as(p).clone()
            optimizer.step()

        torch.testing.assert_close(conv.reshape_as(matrix), matrix)
