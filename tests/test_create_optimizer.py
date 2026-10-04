import warnings

import pytest
import torch

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
