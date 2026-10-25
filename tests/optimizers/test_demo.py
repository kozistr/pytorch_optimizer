from copy import deepcopy

import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer.demo import HAS_EINOPS, DeMo
from tests.utils.distributed import run_distributed


def _check_demo_checkpoint_and_closure(rank):
    params = [
        nn.Parameter(torch.ones(shape, dtype=dtype))
        for shape, dtype in (((4, 4), torch.float32), ((4, 4), torch.bfloat16), ((8, 4), torch.float32))
    ]
    optimizer = DeMo(params, lr=0.01, compression_top_k=1, compression_chunk=4)
    for p in params:
        p.grad = (torch.arange(p.numel()).reshape_as(p).sin() + rank).to(p.dtype)
    optimizer.step()

    restored_params = [nn.Parameter(p.detach().clone()) for p in params]
    restored = DeMo(restored_params, lr=0.01, compression_top_k=1, compression_chunk=4)
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    for step in range(2):
        for current, parameters in ((optimizer, params), (restored, restored_params)):
            current.zero_grad()

            def closure(parameters=parameters, step=step):
                for p in parameters:
                    p.grad = (torch.arange(p.numel()).reshape_as(p).cos() + step + rank).to(p.dtype)
                return 1.0

            if current is optimizer:
                closure()
                current.step()
            else:
                assert current.step(closure) == 1.0
        torch.testing.assert_close(restored_params, params)


@pytest.mark.skipif(not HAS_EINOPS, reason='requires einops')
@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='Gloo is unavailable')
def test_demo_checkpoint_and_closure(tmp_path):
    run_distributed(_check_demo_checkpoint_and_closure, tmp_path / 'store')
