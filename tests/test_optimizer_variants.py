import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.recipes import (
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
        if name != 'nadam'
        for foreach in ([False, True] if name in FOREACH_OPTIMIZERS else [False])
    ],
)
def test_maximize(optimizer_name, foreach):
    optimizer_class = load_optimizer(optimizer_name)
    params = [torch.full((2, 2), 2.0, requires_grad=True) for _ in range(2)]
    params[0].grad = torch.full_like(params[0], 0.5)
    params[1].grad = torch.full_like(params[1], -0.5)

    options = {'foreach': foreach} if optimizer_name in FOREACH_OPTIMIZERS else {}
    if optimizer_name == 'sgd':
        options['lr'] = 1e-3
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


@pytest.mark.parametrize('foreach', [False, True, None])
@pytest.mark.parametrize('adam_debias', [False, True])
@pytest.mark.parametrize('eps', [1e-3, 1e-16])
@pytest.mark.parametrize('gradient_scale', [1.0, 1e-7])
def test_adabelief_bias_corrected_update(foreach, adam_debias, eps, gradient_scale):
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = load_optimizer('adabelief')([param], lr=0.1, eps=eps, foreach=foreach, adam_debias=adam_debias)
    expected = param.detach().clone()
    mean, variance = torch.zeros_like(param), torch.zeros_like(param)

    for step, gradient in enumerate(([0.4, -0.8], [-0.6, 0.1], [0.2, 0.9]), start=1):
        param.grad = torch.tensor(gradient) * gradient_scale
        mean = 0.9 * mean + 0.1 * param.grad
        variance = 0.999 * variance + 0.001 * (param.grad - mean).square() + eps
        denominator = (variance / (1.0 - 0.999 ** step)).sqrt() + eps
        numerator = mean if adam_debias else mean / (1.0 - 0.9 ** step)
        expected -= 0.1 * numerator / denominator

        optimizer.step()
        torch.testing.assert_close(param, expected)


@pytest.mark.parametrize('optimizer_name', ['adabelief', 'lamb'])
@pytest.mark.parametrize('weight_decouple', [False, True])
@pytest.mark.parametrize('maximize', [False, True])
def test_adaptive_foreach_parity(optimizer_name, weight_decouple, maximize):
    scalar_params = [torch.tensor([0.0, 0.0], requires_grad=True), torch.tensor([2.0, -1.0], requires_grad=True)]
    foreach_params = [param.detach().clone().requires_grad_() for param in scalar_params]
    options = {'lr': 0.1, 'weight_decay': 0.1, 'weight_decouple': weight_decouple, 'maximize': maximize}
    scalar = load_optimizer(optimizer_name)(scalar_params, foreach=False, **options)
    batched = load_optimizer(optimizer_name)(foreach_params, foreach=True, **options)

    for gradients in (([0.0, 0.0], [0.4, -0.6]), ([0.2, -0.3], None), ([0.0, 0.0], [-0.1, 0.7])):
        for param, batched_param, gradient in zip(scalar_params, foreach_params, gradients):
            param.grad = torch.tensor(gradient) if gradient is not None else None
            batched_param.grad = param.grad.clone() if param.grad is not None else None

        scalar.step()
        batched.step()

        for param, batched_param in zip(scalar_params, foreach_params):
            torch.testing.assert_close(param, batched_param)


@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('degenerated_to_sgd', [False, True])
def test_lamb_rectification_warmup(foreach, degenerated_to_sgd):
    param = torch.ones(1, requires_grad=True)
    optimizer = load_optimizer('lamb')(
        [param], lr=0.01, rectify=True, degenerated_to_sgd=degenerated_to_sgd, foreach=foreach
    )

    for _ in range(5):
        param.grad = torch.ones_like(param)
        before = param.detach().clone()
        optimizer.step()

        if degenerated_to_sgd:
            assert param < before
        else:
            torch.testing.assert_close(param, before)

    assert optimizer.state[param]['exp_avg'].abs().sum() > 0


def test_lamb_foreach_avoids_scalar_extraction(monkeypatch):
    params = [torch.ones(2, requires_grad=True), torch.tensor([0.0, 0.0], requires_grad=True)]
    optimizer = load_optimizer('lamb')(params, foreach=True, pre_norm=True)
    for param in params:
        param.grad = torch.ones_like(param)

    def reject_scalar_extraction(*_args):
        pytest.fail('Lamb foreach updates must keep scalar values as tensors')

    monkeypatch.setattr(torch.Tensor, 'item', reject_scalar_extraction)
    monkeypatch.setattr(torch.Tensor, '__bool__', reject_scalar_extraction)
    optimizer.step()
