import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.optimizer import (
    BSAM,
    SAM,
    WSAM,
    FriendlySAM,
    Lookahead,
    LookSAM,
    PCGrad,
    Ranger21,
    SafeFP16Optimizer,
    load_optimizer,
)
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector
from tests.constants import PULLBACK_MOMENTUM
from tests.utils import Example, simple_parameter


class TestEpsilonParameters:
    """Tests for epsilon-related parameter validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name'),
        [
            ('Shampoo', 'matrix_eps'),
            ('ScalableShampoo', 'diagonal_eps'),
            ('ScalableShampoo', 'matrix_eps'),
        ],
    )
    def test_shampoo_epsilon_parameters(self, optimizer_name, param_name):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: -1e-6})

    @pytest.mark.parametrize('optimizer_name', ['adafactor', 'came'])
    @pytest.mark.parametrize('param_name', ['eps1', 'eps2'])
    def test_multi_epsilon_parameters(self, optimizer_name, param_name):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: -1e-6})


def test_pcgrad_parameters():
    opt = load_optimizer('adamw')([simple_parameter()])

    for reduction in ('mean', 'sum'):
        PCGrad(opt, reduction=reduction)

    with pytest.raises(ValueError):
        PCGrad(opt, reduction='invalid')


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


@pytest.mark.parametrize('optimizer', [SAM, WSAM, LookSAM, BSAM, FriendlySAM])
def test_sam_family_methods(optimizer):
    base_optimizer = load_optimizer('lion')

    opt = optimizer(params=[simple_parameter()], model=None, base_optimizer=base_optimizer, num_data=1)
    opt.zero_grad()

    opt.init_group({'params': []})
    opt.load_state_dict(opt.state_dict())

    with pytest.raises(NoClosureError):
        opt.step()

    with pytest.raises(ValueError):
        optimizer(model=None, params=None, base_optimizer=base_optimizer, rho=-0.1, num_data=1)


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


class TestRanger21:
    """Tests for Ranger21 optimizer specific functionality."""

    def test_warm_iterations(self):
        assert Ranger21.build_warm_up_iterations(1000, 0.999) == 220
        assert Ranger21.build_warm_up_iterations(4500, 0.999) == 2000
        assert Ranger21.build_warm_down_iterations(1000) == 280

    def test_warm_up_and_down(self):
        lr: float = 1e-1
        opt = Ranger21([simple_parameter(require_grad=False)], num_iterations=500, lr=lr, warm_down_min_lr=3e-5)

        assert opt.warm_up_dampening(lr, 100) == 0.09090909090909091
        assert opt.warm_up_dampening(lr, 200) == 0.1
        assert opt.warm_up_dampening(lr, 300) == 0.1
        assert opt.warm_down(lr, 300) == 0.1
        assert opt.warm_down(lr, 400) == 0.07093070921985817

    def test_closure(self):
        model: nn.Module = Example()
        optimizer = load_optimizer('ranger21')(model.parameters(), num_iterations=100, betas=(0.9, 1e-9))

        loss_fn = nn.BCEWithLogitsLoss()

        def closure():
            loss = loss_fn(torch.ones((1, 1)), model(torch.ones((1, 1))))
            loss.backward()
            return loss

        optimizer.step(closure)


class TestAdaFactorParameters:
    @pytest.mark.parametrize('recipe', [(1.0, 1, True, True, 1e-6), (1.0, 1, False, True, 1.0)])
    def test_get_relative_step_size(self, recipe):
        opt = load_optimizer('adafactor')([simple_parameter(True)])

        expected = opt.get_relative_step_size(*recipe[:-1])

        assert expected == recipe[-1]

    @pytest.mark.parametrize('recipe', [(1.0, 1.0, True, 1.0), (2.0, 1.0, False, 2.0)])
    def test_get_lr(self, recipe):
        opt = load_optimizer('adafactor')([simple_parameter(True)])

        expected = opt.get_lr(*recipe[:-1])

        assert expected == recipe[-1]


class TestA2GradParameters:
    """Tests for A2Grad optimizer parameter validation."""

    @pytest.mark.parametrize(
        ('param_name', 'param_value'),
        [
            ('lips', -1.0),
            ('rho', -0.1),
        ],
    )
    def test_negative_parameters(self, param_name, param_value):
        param = [simple_parameter(require_grad=False)]
        opt = load_optimizer('a2grad')
        params = param if param_name == 'lips' else None
        with pytest.raises(ValueError):
            opt(params, **{param_name: param_value})

    @pytest.mark.parametrize('variant', ['uni', 'inc', 'exp'])
    def test_valid_variants(self, variant):
        param = [simple_parameter(require_grad=False)]
        opt = load_optimizer('a2grad')
        opt(param, variant=variant)

    def test_invalid_variant(self):
        param = [simple_parameter(require_grad=False)]
        opt = load_optimizer('a2grad')
        with pytest.raises(ValueError):
            opt(param, variant='dummy')


def test_amos_get_scale():
    opt = load_optimizer('amos')

    assert opt.get_scale(torch.zeros((1,))) == 0.5
    assert opt.get_scale(torch.zeros((1, 4))) == 0.7071067811865476
    assert opt.get_scale(torch.zeros((1, 16, 2, 2))) == 0.25


class TestNegativeParameterValidation:
    """Tests for negative parameter value validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name', 'param_value'),
        [
            ('accsgd', 'xi', -0.1),
            ('accsgd', 'kappa', -0.1),
            ('asgd', 'amplifier', -1.0),
            ('lars', 'dampening', -0.1),
            ('lars', 'trust_coefficient', -1e-3),
            ('ranger', 'alpha', -0.1),
            ('ranger', 'k', -1),
        ],
    )
    def test_negative_parameter_raises_error(self, optimizer_name, param_name, param_value):
        opt = load_optimizer(optimizer_name)
        params = [simple_parameter(False)] if optimizer_name in ('accsgd', 'asgd') else None
        with pytest.raises(ValueError):
            opt(params, **{param_name: param_value})

    def test_accsgd_constant_validation(self):
        opt = load_optimizer('accsgd')
        with pytest.raises(ValueError):
            opt([simple_parameter(False)], constant=42)


class TestInvalidOptionParameters:
    """Tests for invalid option/enum parameter validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name', 'invalid_value'),
        [
            ('apollodqn', 'rebound', 'dummy'),
            ('apollodqn', 'weight_decay_type', 'dummy'),
        ],
    )
    def test_invalid_option_raises_error(self, optimizer_name, param_name, invalid_value):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: invalid_value})


class TestGaLoreProjector:
    """Tests for GaLore projector methods."""

    @pytest.fixture
    def sample_tensor(self):
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    def test_invalid_projection_type_project_with_ortho(self, sample_tensor):
        invalid_galore = GaLoreProjector(projection_type='invalid')
        invalid_galore.ortho_matrix = sample_tensor
        with pytest.raises(NotImplementedError):
            invalid_galore.project(sample_tensor, 1)

    def test_invalid_projection_type_project(self, sample_tensor):
        with pytest.raises(NotImplementedError):
            GaLoreProjector(projection_type='invalid').project(sample_tensor, 1)

    def test_invalid_projection_type_project_back(self, sample_tensor):
        with pytest.raises(NotImplementedError):
            GaLoreProjector(projection_type='invalid').project_back(sample_tensor)

    def test_full_projection_project_validation(self, sample_tensor):
        full_projector = GaLoreProjector(projection_type='full')
        full_projector.ortho_matrix = sample_tensor
        with pytest.raises(ValueError):
            full_projector.project(sample_tensor, 1)

    def test_full_projection_project_back_validation(self, sample_tensor):
        full_projector = GaLoreProjector(projection_type='full')
        full_projector.ortho_matrix = sample_tensor
        with pytest.raises(ValueError):
            full_projector.project_back(sample_tensor)

    def test_left_projection_with_random_matrix(self, sample_tensor):
        projector = GaLoreProjector(projection_type='left', rank=1)
        projector.get_orthogonal_matrix(sample_tensor, projection_type='left', from_random_matrix=True)

    def test_left_projection_without_rank(self, sample_tensor):
        projector = GaLoreProjector(projection_type='left', rank=None)
        with pytest.raises(TypeError):
            projector.get_orthogonal_matrix(sample_tensor, projection_type='left', from_random_matrix=True)

    def test_std_projection_invalid(self, sample_tensor):
        projector = GaLoreProjector(projection_type='std', rank=1)
        with pytest.raises(ValueError):
            projector.get_orthogonal_matrix(sample_tensor, projection_type='std')


@pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO'])
def test_muon_use_muon_param(optimizer_name):
    with pytest.raises(ValueError):
        load_optimizer(optimizer_name)([Example().parameters()])
