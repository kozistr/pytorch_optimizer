import pytest
import torch

from tests.constants import (
    ADAMD_SUPPORTED_OPTIMIZERS,
    ADANORM_SUPPORTED_OPTIMIZERS,
    COPT_SUPPORTED_OPTIMIZERS,
    STABLE_ADAMW_SUPPORTED_OPTIMIZERS,
)
from tests.utils import TrainingRunner, build_model, build_optimizer_parameter, ids, simple_parameter


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_optimizer(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config, adanorm=True)

    runner = TrainingRunner(model, loss_fn, optimizer, x_data, y_data)
    runner.run(iterations=num_iterations, threshold=1.75)


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
    model, loss_fn = build_model()

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config, adam_debias=True)

    create_graph = optimizer_class.__name__ in ('AdaHessian',)
    runner = TrainingRunner(model, loss_fn, optimizer, x_data, y_data)
    runner.run(iterations=num_iterations, create_graph=create_graph, threshold=2.0)


@pytest.mark.parametrize('optimizer_config', COPT_SUPPORTED_OPTIMIZERS, ids=ids)
def test_cautious_variant(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer_class, config, num_iterations = optimizer_config
    parameters, config = build_optimizer_parameter(model.parameters(), optimizer_class.__name__, config)
    optimizer = optimizer_class(parameters, **config, cautious=True)

    runner = TrainingRunner(model, loss_fn, optimizer, x_data, y_data)
    runner.run(iterations=num_iterations, threshold=1.5)


@pytest.mark.parametrize('optimizer_config', STABLE_ADAMW_SUPPORTED_OPTIMIZERS, ids=ids)
def test_stable_adamw_variant(optimizer_config, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer_class, config, num_iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    runner = TrainingRunner(model, loss_fn, optimizer, x_data, y_data)
    runner.run(iterations=num_iterations, threshold=1.5)
