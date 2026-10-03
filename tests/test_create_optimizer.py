import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.optim.lr_scheduler import StepLR

from pytorch_optimizer.optimizer import create_optimizer, load_optimizer
from tests.constants import COMPILE_SUPPORTED_OPTIMIZERS, SKIP_CREATE_OPTIMIZER, VALID_OPTIMIZER_NAMES
from tests.utils import Example, Trainer, build_model, ids

WRAPPER_TEST_OPTIMIZERS = ['adamp', 'lion', 'lamb', 'adan', 'madgrad', 'ranger']


def _get_optimizer_kwargs(optimizer_name):
    kwargs = {'eps': 1e-8, 'k': 7}
    if optimizer_name == 'ranger21':
        kwargs['num_iterations'] = 1
    elif optimizer_name == 'bsam':
        kwargs['num_data'] = 1
    return kwargs


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_create_optimizer_basic(optimizer_name):
    if optimizer_name in SKIP_CREATE_OPTIMIZER:
        pytest.skip(f'skip {optimizer_name}')

    optimizer = create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=False,
        use_orthograd=False,
        **_get_optimizer_kwargs(optimizer_name),
    )
    assert optimizer.defaults.get('weight_decay', 0.0) == 0.0
    assert all(group.get('weight_decay', 0.0) == 0.0 for group in optimizer.param_groups)


@pytest.mark.parametrize('optimizer_name', WRAPPER_TEST_OPTIMIZERS)
def test_create_optimizer_with_lookahead(optimizer_name):
    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=True,
        use_orthograd=False,
        **_get_optimizer_kwargs(optimizer_name),
    )


@pytest.mark.parametrize('optimizer_config', COMPILE_SUPPORTED_OPTIMIZERS, ids=ids)
@pytest.mark.parametrize('foreach', [False, True])
def test_create_compiled_optimizer(optimizer_config, foreach, environment):
    torch._dynamo.reset()
    counter = CompileCounterWithBackend('aot_eager')
    optimizer_class, config, iterations = optimizer_config
    config = config.copy()
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)
    lr = torch.tensor(config.pop('lr'), device=x_data.device)
    if optimizer_class.__name__ == 'AdamW':
        config['capturable'] = foreach

    optimizer = create_optimizer(
        model,
        optimizer_class.__name__,
        lr=lr,
        foreach=foreach,
        compile_step=True,
        compile_kwargs={'backend': counter},
        **config,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    initial_loss = trainer.compute_loss()

    for step in range(iterations):
        optimizer.zero_grad()
        trainer.compute_loss().backward()
        optimizer.step()
        scheduler.step()
        if step == 1:
            frame_count = counter.frame_count
        elif step > 1:
            assert counter.frame_count == frame_count

    assert frame_count > 0
    trainer.assert_loss_decreased(initial_loss, trainer.compute_loss(), threshold=1.5)
    explanation = torch._dynamo.explain(optimizer_class.step)(optimizer)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    torch._dynamo.reset()


@pytest.mark.parametrize('optimizer_name', WRAPPER_TEST_OPTIMIZERS)
def test_create_optimizer_with_orthograd(optimizer_name):
    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=False,
        use_orthograd=True,
        **_get_optimizer_kwargs(optimizer_name),
    )


@pytest.mark.parametrize(
    ('optimizer_name', 'package_flag'),
    [
        ('bnb_adamw8bit', 'HAS_BNB'),
        ('q_galore_adamw8bit', 'HAS_Q_GALORE'),
        ('torchao_adamw4bit', 'HAS_TORCHAO'),
    ],
)
def test_external_optimizers_require_import(optimizer_name, package_flag, monkeypatch):
    monkeypatch.setattr(f'pytorch_optimizer.optimizer.{package_flag}', False)
    with pytest.raises(ImportError):
        load_optimizer(optimizer_name)
