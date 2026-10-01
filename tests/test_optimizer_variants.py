import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.constants import (
    ADAMD_SUPPORTED_OPTIMIZERS,
    ADANORM_SUPPORTED_OPTIMIZERS,
    COPT_SUPPORTED_OPTIMIZERS,
    FOREACH_OPTIMIZERS,
    MAXIMIZE_OPTIMIZERS,
    STABLE_ADAMW_SUPPORTED_OPTIMIZERS,
)
from tests.utils import Trainer, build_model, build_optimizer_parameter, ids, simple_parameter


@pytest.mark.parametrize(
    ('optimizer_name', 'foreach'),
    [
        (name, foreach)
        for name in sorted(MAXIMIZE_OPTIMIZERS)
        for foreach in ([False, True] if name in FOREACH_OPTIMIZERS else [False])
    ],
)
def test_maximize(optimizer_name, foreach):
    optimizer_class = load_optimizer(optimizer_name)
    params = [torch.full((2, 2), 2.0, requires_grad=True) for _ in range(2)]
    params[0].grad = torch.full_like(params[0], 0.5)
    params[1].grad = torch.full_like(params[1], -0.5)

    options = {'foreach': foreach} if optimizer_name in FOREACH_OPTIMIZERS else {}
    ascent = optimizer_class([params[0]], maximize=True, **options)
    descent = optimizer_class([params[1]], maximize=False, **options)

    if optimizer_name.startswith('schedulefree'):
        ascent.train()
        descent.train()

    ascent.step()
    descent.step()

    torch.testing.assert_close(params[0], params[1])


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_optimizer(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config, adanorm=True)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=num_iterations, threshold=1.75)


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_variant(optimizer_config):
    param = simple_parameter(True)
    param.grad = torch.ones(1, 1)

    optimizer_class, _ = optimizer_config[:2]

    optimizer = optimizer_class([param], adanorm=True)
    optimizer.step()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('optimizer_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adamd_variant(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config, adam_debias=True)

    create_graph = optimizer_class.__name__ in ('AdaHessian',)
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=num_iterations, create_graph=create_graph, threshold=2.0)


@pytest.mark.parametrize('optimizer_config', COPT_SUPPORTED_OPTIMIZERS, ids=ids)
def test_cautious_variant(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer_class, config, num_iterations = optimizer_config
    parameters, config = build_optimizer_parameter(model.parameters(), optimizer_class.__name__, config)
    optimizer = optimizer_class(parameters, **config, cautious=True)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=num_iterations, threshold=1.5)


@pytest.mark.parametrize('optimizer_config', STABLE_ADAMW_SUPPORTED_OPTIMIZERS, ids=ids)
def test_stable_adamw_variant(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=num_iterations, threshold=1.5)
