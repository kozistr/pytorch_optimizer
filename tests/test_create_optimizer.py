import warnings
from copy import deepcopy

import pytest
import torch
from torch._dynamo.testing import CompileCounter

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
    kwargs = {'eps': 1e-8, 'k': 7}
    if optimizer_name == 'ranger21':
        kwargs['num_iterations'] = 1
    elif optimizer_name == 'bsam':
        kwargs['num_data'] = 1
    return kwargs


class TestCreateOptimizer:
    @pytest.mark.parametrize(
        'optimizer_name',
        [name for name in VALID_OPTIMIZER_NAMES if name not in SKIP_CREATE_OPTIMIZER],
    )
    def test_create_optimizer_basic(self, optimizer_name):
        optimizer = create_optimizer(
            TrainingModel(),
            optimizer_name=optimizer_name,
            use_lookahead=False,
            use_orthograd=False,
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
            use_orthograd=False,
            **_get_optimizer_kwargs(optimizer_name),
        )

        assert isinstance(optimizer, Lookahead) == (optimizer_name == 'adamp')

    def test_create_optimizer_with_orthograd(self):
        optimizer = create_optimizer(
            TrainingModel(),
            optimizer_name='adamp',
            use_lookahead=False,
            use_orthograd=True,
            **_get_optimizer_kwargs('adamp'),
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

        if optimizer_name == 'adamw':
            config['capturable'] = foreach

        optimizer = create_optimizer(
            model,
            optimizer_name,
            foreach=foreach,
            compile=True,
            compile_kwargs={'backend': 'aot_eager'},
            **config,
        )

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=iterations)

        if optimizer_name == 'yogi' and foreach:
            model = TrainingModel(dtype=torch.bfloat16)
            optimizer = create_optimizer(
                model, 'yogi', lr=0.01, betas=(0.5, 0.5), initial_accumulator=0.494140625,
                foreach=True, compile=True, compile_kwargs={'disable': True},
            )

            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.703125)

            optimizer.step()

            # Rounded gradient squares match the stored moments, so their sign updates are zero.
            for parameter in model.parameters():
                torch.testing.assert_close(
                    optimizer.state[parameter]['exp_avg_sq'], torch.full_like(parameter, 0.494140625)
                )

    @pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
    def test_compiled_foreach_scheduler_and_checkpoint(self):
        torch._dynamo.reset()

        model = TrainingModel()
        lr = torch.tensor(0.1)
        counter = CompileCounter()
        optimizer = create_optimizer(
            model, 'radam', lr=lr, betas=(0.6, 0.8), foreach=True, compile=True,
            degenerated_to_sgd=True, compile_kwargs={'backend': counter, 'fullgraph': True},
        )

        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.1)

        for _ in range(12):
            lr.mul_(0.99)
            expected_lr = lr.clone()
            optimizer.step()

            torch.testing.assert_close(lr, expected_lr)

        assert counter.frame_count == 2
        assert optimizer.param_groups[0]['step'] == 12

        checkpoint = deepcopy(optimizer.state_dict())
        optimizer.step()
        optimizer.load_state_dict(checkpoint)
        optimizer.step()

        assert optimizer.param_groups[0]['step'] == 13
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


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
