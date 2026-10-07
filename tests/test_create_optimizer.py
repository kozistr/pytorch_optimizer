import warnings
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.optimizer import Lookahead, OrthoGrad, create_optimizer, load_optimizer
from tests.fixtures import TrainingModel, build_model
from tests.optimizer_cases import SKIP_CREATE_OPTIMIZER, VALID_OPTIMIZER_NAMES
from tests.recipes import COMPILE_SUPPORTED_OPTIMIZERS
from tests.utils import Trainer, ids


@pytest.fixture(autouse=True)
def ignore_factory_warnings():
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', ImportWarning)


def _get_optimizer_kwargs(optimizer_name):
    kwargs = {}
    if optimizer_name == 'ranger21':
        kwargs['num_iterations'] = 1
    elif optimizer_name == 'bsam':
        kwargs['num_data'] = 1
    return kwargs


class TestCreateOptimizer:
    @pytest.mark.parametrize('optimizer_name', ['muon', 'adamuon'])
    def test_muon_tied_parameters_and_head(self, optimizer_name):
        model = nn.Module()
        model.body = nn.Linear(2, 2)
        model.frozen = nn.Linear(2, 2).requires_grad_(False)
        model.embedding = nn.Embedding(4, 2)
        model.projection = nn.Linear(2, 4, bias=False)
        model.projection.weight = model.embedding.weight
        model.head = nn.Linear(2, 1, bias=False)

        optimizer = create_optimizer(model, optimizer_name)
        groups = {group['use_muon']: group['params'] for group in optimizer.param_groups}

        assert [id(p) for p in groups[True]] == [id(model.body.weight)]
        assert {id(p) for p in groups[False]} == {
            id(model.body.bias), id(model.embedding.weight), id(model.head.weight)
        }

    @pytest.mark.parametrize(
        'optimizer_name',
        [name for name in VALID_OPTIMIZER_NAMES if name not in SKIP_CREATE_OPTIMIZER],
    )
    def test_create_optimizer_basic(self, optimizer_name):
        optimizer = create_optimizer(
            TrainingModel(),
            optimizer_name=optimizer_name,
            **_get_optimizer_kwargs(optimizer_name),
        )

        assert optimizer.defaults.get('weight_decay', 0.0) == 0.0
        assert all(group.get('weight_decay', 0.0) == 0.0 for group in optimizer.param_groups)

    @pytest.mark.parametrize('optimizer_name', ['adamp', 'ranger', 'ranger21', 'ranger25'])
    def test_create_optimizer_with_lookahead(self, optimizer_name):
        optimizer = create_optimizer(
            TrainingModel(),
            optimizer_name=optimizer_name,
            use_lookahead=True,
            **_get_optimizer_kwargs(optimizer_name),
        )

        assert isinstance(optimizer, Lookahead) == (optimizer_name == 'adamp')

    def test_create_optimizer_with_orthograd(self):
        optimizer = create_optimizer(
            TrainingModel(),
            optimizer_name='adamp',
            use_orthograd=True,
        )

        assert isinstance(optimizer, OrthoGrad)

    @pytest.mark.parametrize('optimizer_config', COMPILE_SUPPORTED_OPTIMIZERS, ids=ids)
    @pytest.mark.parametrize('foreach', [False, True])
    @pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
    def test_create_compiled_optimizer(self, optimizer_config, foreach, environment):
        torch._dynamo.reset()

        optimizer_name, config, iterations = optimizer_config
        config = config.copy()

        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)
        counter = CompileCounterWithBackend('aot_eager')

        if optimizer_name == 'adamw':
            config['capturable'] = foreach

        optimizer = create_optimizer(
            model,
            optimizer_name,
            foreach=foreach,
            compile=True,
            compile_kwargs={'backend': counter},
            **config,
        )

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=iterations)

        if foreach and isinstance(optimizer, BaseOptimizer) and optimizer._supports_compiled_foreach:
            lr = torch.tensor(config['lr'], device=x_data.device)
            for group in optimizer.param_groups:
                group['lr'] = lr

            for _ in range(3):
                lr.mul_(0.99)
                expected_lr = lr.clone()
                frames = counter.frame_count
                optimizer.step()

                torch.testing.assert_close(lr, expected_lr)

            assert 0 < counter.frame_count == frames

    @pytest.mark.parametrize('foreach', [False, True])
    def test_create_optimizer_compile_wiring(self, foreach):
        model = TrainingModel(dtype=torch.bfloat16)

        with patch('torch.compile', side_effect=lambda step, **_: step) as compile_step:
            optimizer = create_optimizer(
                model, 'yogi', lr=0.01, betas=(0.5, 0.5), initial_accumulator=0.494140625,
                foreach=foreach, compile=True, compile_kwargs={'dynamic': False},
            )

        step = optimizer._apply_update_foreach.__wrapped__ if foreach else optimizer.step.__func__
        compile_step.assert_called_once_with(step, dynamic=False)

        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.703125)

        for lr in (0.01, 0.02, torch.tensor(0.03)):
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()

            # Rounded gradient squares match the stored moments, so their sign updates are zero.
            for parameter in model.parameters():
                torch.testing.assert_close(
                    optimizer.state[parameter]['exp_avg_sq'], torch.full_like(parameter, 0.494140625)
                )

    @pytest.mark.parametrize(
        ('optimizer_name', 'options'),
        [
            ('lion', {}),
            ('tiger', {}),
            ('signsgd', {'momentum': 0.0}),
            ('signsgd', {'momentum': 0.9}),
            ('sgdw', {'momentum': 0.0}),
            ('sgdw', {'momentum': 0.9, 'dampening': 0.2}),
            ('sgdw', {'momentum': 0.9, 'nesterov': True}),
        ],
    )
    @pytest.mark.parametrize('weight_decouple', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    @pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
    def test_compiled_foreach_updates(self, optimizer_name, options, weight_decouple, dtype):
        torch._dynamo.reset()

        model = TrainingModel(dtype=dtype)
        model.fc2.to(dtype=torch.float64)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2.0)
        reference_model = deepcopy(model)

        counter = CompileCounterWithBackend('aot_eager')
        config = {'lr': 0.1, 'weight_decay': 0.2, 'weight_decouple': weight_decouple, 'maximize': True, **options}
        optimizer = create_optimizer(
            model, optimizer_name, compile=True, compile_kwargs={'backend': counter, 'fullgraph': True}, **config
        )
        reference = create_optimizer(reference_model, optimizer_name, foreach=False, **config)

        assert all(group['foreach'] for group in optimizer.param_groups)

        for iteration, lr in enumerate((0.1, 0.05, torch.tensor(0.025), 0.0125)):
            for candidate in (optimizer, reference):
                for group in candidate.param_groups:
                    group['lr'] = lr

            frames = counter.frame_count
            for index, (parameter, expected) in enumerate(zip(model.parameters(), reference_model.parameters())):
                gradient = torch.full_like(parameter, (-1.0) ** iteration * (index + 1) / 4.0)
                parameter.grad = gradient.clone() if index != 1 else None
                expected.grad = gradient.clone() if index != 1 else None

            optimizer.step()
            reference.step()

            for parameter, expected in zip(model.parameters(), reference_model.parameters()):
                torch.testing.assert_close(parameter, expected)
                actual_state, expected_state = optimizer.state[parameter], reference.state[expected]
                assert actual_state.keys() == expected_state.keys()
                for key in expected_state:
                    torch.testing.assert_close(actual_state[key], expected_state[key])

            if iteration > 1:
                assert 0 < counter.frame_count == frames

    @pytest.mark.parametrize(
        ('optimizer_name', 'options'),
        [
            ('muon', {}),
            ('muon', {'nesterov': False}),
            ('muon', {'use_adjusted_lr': True}),
            ('adamuon', {}),
            ('adamuon', {'use_adjusted_lr': True}),
        ],
    )
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    @pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
    def test_compiled_muon_updates(self, optimizer_name, options, dtype, device):
        torch._dynamo.reset()

        model = TrainingModel(dtype=dtype, output_features=2).to(device)
        model.conv = nn.Conv2d(2, 2, 2, bias=False, dtype=dtype, device=device)
        model.projection = nn.Linear(2, 8, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2.0)
        reference_model = deepcopy(model)

        counter = CompileCounterWithBackend('aot_eager')
        config = {'lr': 0.1, 'adamw_lr': 0.01, 'weight_decay': 0.2, 'adamw_wd': 0.1, 'ns_steps': 3, **options}
        optimizer = create_optimizer(
            model, optimizer_name, compile=True, compile_kwargs={'backend': counter, 'fullgraph': True}, **config
        )
        reference = create_optimizer(reference_model, optimizer_name, foreach=False, **config)

        assert all(group['foreach'] for group in optimizer.param_groups)
        assert {group['use_muon'] for group in optimizer.param_groups} == {False, True}

        for iteration, lr in enumerate((0.1, 0.05, torch.tensor(0.025, device=device), 0.0125)):
            expected_lr = lr.clone() if isinstance(lr, torch.Tensor) else lr
            for candidate in (optimizer, reference):
                for group in candidate.param_groups:
                    group['lr'] = lr if group['use_muon'] else lr / 10.0

            for parameter, expected in zip(model.parameters(), reference_model.parameters()):
                if parameter.ndim >= 2:
                    gradient = torch.eye(parameter.size(0), parameter.numel() // parameter.size(0), device=device)
                    gradient[1].mul_(0.5)
                    gradient = gradient.reshape(parameter.shape).to(dtype)
                else:
                    gradient = torch.ones_like(parameter)
                parameter.grad = gradient.mul((-1.0) ** iteration)
                expected.grad = parameter.grad.clone()

            frames = counter.frame_count
            optimizer.step()
            reference.step()

            torch.testing.assert_close(lr, expected_lr)
            for parameter, expected in zip(model.parameters(), reference_model.parameters()):
                torch.testing.assert_close(parameter, expected)
                for key in reference.state[expected]:
                    torch.testing.assert_close(optimizer.state[parameter][key], reference.state[expected][key])

            if iteration > 0:
                assert 0 < counter.frame_count == frames


class TestOptionalIntegrations:
    @pytest.mark.parametrize(
        ('optimizer_name', 'package_flag'),
        [
            ('bnb_adamw8bit', 'HAS_BNB'),
            ('q_galore_adamw8bit', 'HAS_Q_GALORE'),
            ('torchao_adamw4bit', 'HAS_TORCHAO'),
        ],
    )
    def test_external_optimizers_require_import(self, optimizer_name, package_flag, monkeypatch):
        monkeypatch.setattr(f'pytorch_optimizer.optimizer.{package_flag}', False)
        with pytest.raises(ImportError):
            load_optimizer(optimizer_name)
