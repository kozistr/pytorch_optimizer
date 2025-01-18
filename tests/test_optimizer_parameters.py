import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import (
    SAM,
    TRAC,
    WSAM,
    Lookahead,
    OrthoGrad,
    PCGrad,
    Ranger21,
    SafeFP16Optimizer,
    load_optimizer,
)
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector
from tests.constants import PULLBACK_MOMENTUM
from tests.utils import Example, simple_parameter, simple_zero_rank_parameter


def test_shampoo_parameters():
    with pytest.raises(ValueError):
        load_optimizer('Shampoo')(None, matrix_eps=-1e-6)


def test_scalable_shampoo_parameters():
    opt = load_optimizer('ScalableShampoo')

    with pytest.raises(ValueError):
        opt(None, diagonal_eps=-1e-6)

    with pytest.raises(ValueError):
        opt(None, matrix_eps=-1e-6)


def test_adafactor_parameters():
    opt = load_optimizer('adafactor')

    with pytest.raises(ValueError):
        opt(None, eps1=-1e-6)

    with pytest.raises(ValueError):
        opt(None, eps2=-1e-6)


def test_came_parameters():
    opt = load_optimizer('came')

    with pytest.raises(ValueError):
        opt(None, eps1=-1e-6)

    with pytest.raises(ValueError):
        opt(None, eps2=-1e-6)


def test_pcgrad_parameters():
    opt = load_optimizer('adamp')([simple_parameter()])

    # test reduction
    for reduction in ['mean', 'sum']:
        PCGrad(opt, reduction=reduction)

    with pytest.raises(ValueError):
        PCGrad(opt, reduction='wrong')


def test_sam_parameters():
    with pytest.raises(ValueError):
        SAM(None, load_optimizer('adamp'), rho=-0.1)


def test_wsam_parameters():
    with pytest.raises(ValueError):
        WSAM(None, None, load_optimizer('adamp'), rho=-0.1)


def test_lookahead_parameters():
    optimizer_instance = load_optimizer('adamp')
    optimizer = optimizer_instance([simple_parameter()])

    for pullback_momentum in PULLBACK_MOMENTUM:
        opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
        opt.load_state_dict(opt.state_dict())

    opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
    opt.backup_and_load_cache()
    opt.clear_and_load_backup()

    _ = opt.__getstate__()

    with pytest.raises(ValueError):
        Lookahead(optimizer, k=0)

    with pytest.raises(ValueError):
        Lookahead(optimizer, alpha=-0.1)

    with pytest.raises(ValueError):
        Lookahead(optimizer, pullback_momentum='invalid')


def test_sam_methods():
    optimizer = SAM([simple_parameter()], load_optimizer('adamp'))
    optimizer.reset()
    optimizer.load_state_dict(optimizer.state_dict())


def test_wsam_methods():
    optimizer = WSAM(None, [simple_parameter()], load_optimizer('adamp'))
    optimizer.reset()
    optimizer.load_state_dict(optimizer.state_dict())


def test_safe_fp16_methods():
    optimizer = SafeFP16Optimizer(load_optimizer('adamp')([simple_parameter()], lr=5e-1))
    optimizer.load_state_dict(optimizer.state_dict())
    optimizer.scaler.decrease_loss_scale()
    optimizer.zero_grad()
    optimizer.update_main_grads()
    optimizer.clip_main_grads(100.0)
    optimizer.multiply_grads(100.0)

    with pytest.raises(AttributeError):
        optimizer.get_lr()

    with pytest.raises(AttributeError):
        optimizer.set_lr(lr=5e-1)

    assert optimizer.loss_scale == 2.0 ** (15 - 1)


def test_ranger21_warm_iterations():
    assert Ranger21.build_warm_up_iterations(1000, 0.999) == 220
    assert Ranger21.build_warm_up_iterations(4500, 0.999) == 2000
    assert Ranger21.build_warm_down_iterations(1000) == 280


def test_ranger21_warm_up_and_down():
    lr: float = 1e-1
    opt = Ranger21([simple_parameter(require_grad=False)], num_iterations=500, lr=lr, warm_down_min_lr=3e-5)

    assert opt.warm_up_dampening(lr, 100) == 0.09090909090909091
    assert opt.warm_up_dampening(lr, 200) == 0.1
    assert opt.warm_up_dampening(lr, 300) == 0.1
    assert opt.warm_down(lr, 300) == 0.1
    assert opt.warm_down(lr, 400) == 0.07093070921985817


def test_ranger21_closure():
    model: nn.Module = Example()
    optimizer = load_optimizer('ranger21')(model.parameters(), num_iterations=100, betas=(0.9, 1e-9))

    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        loss = loss_fn(torch.ones((1, 1)), model(torch.ones((1, 1))))
        loss.backward()
        return loss

    optimizer.step(closure)


def test_adafactor_reset():
    opt = load_optimizer('adafactor')([simple_zero_rank_parameter(True)])
    opt.reset()


def test_adafactor_get_lr():
    model: nn.Module = Example()
    opt = load_optimizer('adafactor')(model.parameters())

    recipes = [(True, 1e-6), (False, 1e-2)]

    for warmup_init, expected_lr in recipes:
        assert opt.get_lr(1.0, 1, 1.0, True, warmup_init, True) == expected_lr


def test_came_reset():
    opt = load_optimizer('came')([simple_zero_rank_parameter(True)])
    opt.reset()


def test_a2grad_parameters():
    param = [simple_parameter(require_grad=False)]
    opt = load_optimizer('a2grad')

    # test lipschitz constant
    with pytest.raises(ValueError):
        opt(param, lips=-1.0)

    # test rho
    with pytest.raises(ValueError):
        opt(None, rho=-0.1)

    # test variants
    for variant in ['uni', 'inc', 'exp']:
        opt(param, variant=variant)

    with pytest.raises(ValueError):
        opt(param, variant='dummy')


def test_amos_get_scale():
    opt = load_optimizer('amos')

    assert opt.get_scale(torch.zeros((1,))) == 0.5
    assert opt.get_scale(torch.zeros((1, 4))) == 0.7071067811865476
    assert opt.get_scale(torch.zeros((1, 16, 2, 2))) == 0.25


def test_accsgd_parameters():
    param = [simple_parameter(False)]
    opt = load_optimizer('accsgd')

    with pytest.raises(ValueError):
        opt(param, xi=-0.1)

    with pytest.raises(ValueError):
        opt(param, kappa=-0.1)

    with pytest.raises(ValueError):
        opt(param, constant=42)


def test_asgd_parameters():
    opt = load_optimizer('asgd')

    # test amplifier
    with pytest.raises(ValueError):
        opt([simple_parameter(False)], amplifier=-1.0)


def test_lars_parameters():
    opt = load_optimizer('lars')

    # test dampening
    with pytest.raises(ValueError):
        opt(None, dampening=-0.1)

    # test trust_coefficient
    with pytest.raises(ValueError):
        opt(None, trust_coefficient=-1e-3)


def test_apollo_parameters():
    opt = load_optimizer('apollodqn')

    # test rebound type
    with pytest.raises(ValueError):
        opt(None, rebound='dummy')

    # test weight_decay_type
    with pytest.raises(ValueError):
        opt(None, weight_decay_type='dummy')


def test_ranger_parameters():
    opt = load_optimizer('ranger')

    # test ema ratio `alpha`
    with pytest.raises(ValueError):
        opt(None, alpha=-0.1)

    # test lookahead step `k`
    with pytest.raises(ValueError):
        opt(None, k=-1)


def test_galore_projection_type():
    p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    _ = GaLoreProjector.get_orthogonal_matrix(p, 1, projection_type='left', from_random_matrix=True)

    with pytest.raises(NotImplementedError):
        GaLoreProjector(projection_type='invalid').project(p, 1)

    with pytest.raises(NotImplementedError):
        GaLoreProjector(projection_type='invalid').project_back(p)

    with pytest.raises(ValueError):
        GaLoreProjector.get_orthogonal_matrix(p, 1, projection_type='std')


@pytest.mark.parametrize('optimizer_instance', [Lookahead, OrthoGrad, TRAC])
def test_load_optimizer(optimizer_instance):
    params = [simple_parameter()]

    _ = optimizer_instance(torch.optim.AdamW(params))
    _ = optimizer_instance(torch.optim.AdamW, params=params)

    with pytest.raises(ValueError):
        optimizer_instance(torch.optim.AdamW)
