from typing import Any

import pytest
import torch

from tests.fixtures import build_model, make_parameter
from tests.optimizer_cases import FOREACH_OPTIMIZERS, GRADIENT_OPTIONS, MAXIMIZE_OPTIMIZERS, SKIP_CAPABILITY_PROBE
from tests.utils import Trainer, build_optimizer, build_optimizer_parameters, ids

ADANORM_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adabelief', {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    ('adamp', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('adams', {'lr': 7.5e-1, 'weight_decay': 1e-3}, 5),
    ('adapnm', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('diffgrad', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('lamb', {'lr': 5e-2}, 10),
    ('radam', {'lr': 5e0, 'weight_decay': 1e-3}, 10),
    ('ranger', {'lr': 2.5e0, 'weight_decay': 1e-3}, 75),
    ('adan', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('lion', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('adamax', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('aida', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
]

ADAMD_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adabelief', {'lr': 1e1, 'weight_decay': 1e-3}, 5),
    ('adabound', {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3}, 35),
    ('adamp', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adams', {'lr': 2e0, 'weight_decay': 1e-3}, 5),
    ('diffgrad', {'lr': 2e0, 'weight_decay': 1e-3, 'rectify': True}, 15),
    ('diffgrad', {'lr': 2e0, 'weight_decay': 1e-3}, 5),
    ('lamb', {'lr': 1e0, 'weight_decay': 1e-3, 'rectify': True}, 30),
    ('radam', {'lr': 1e0, 'weight_decay': 1e-3}, 25),
    ('ranger', {'lr': 5e0, 'weight_decay': 1e-3}, 50),
    ('ranger21', {'lr': 5e-1, 'weight_decay': 1e-3, 'num_iterations': 125, 'disable_lr_scheduler': True}, 125),
    ('adapnm', {'lr': 1e0, 'weight_decay': 1e-3}, 10),
    ('novograd', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adanorm', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('yogi', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adamod', {'lr': 1e2, 'weight_decay': 1e-3}, 20),
    ('adamax', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('avagrad', {'lr': 1e1, 'weight_decay': 1e-3}, 5),
    ('adahessian', {'lr': 5e0, 'weight_decay': 1e-3}, 5),
    ('aida', {'lr': 1e1, 'weight_decay': 1e-3, 'rectify': True}, 10),
]

COPT_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adafactor', {'lr': 1e0, 'weight_decay': 1e-3, 'scale_parameter': False, 'relative_step': False}, 5),
    ('lion', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('ademamix', {'lr': 1e0}, 2),
    ('laprop', {'lr': 1e0}, 2),
    ('adamp', {'lr': 1e0}, 2),
    ('adopt', {'lr': 1e1}, 3),
    ('adashift', {'lr': 1e0, 'keep_num': 1}, 5),
    ('mars', {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3}, 3),
    ('mars', {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'optimize_1d': True}, 3),
    (
        'muon',
        {
            'lr': 5e-1,
            'weight_decay': 1e-3,
            'use_adjusted_lr': True,
            'adamw_lr': 5e-1,
            'adamw_betas': (0.9, 0.98),
            'adamw_wd': 1e-2,
        },
        7,
    ),
    (
        'adago',
        {'lr': 5e-1, 'use_adjusted_lr': True, 'adamw_lr': 5e-1, 'adamw_betas': (0.9, 0.98), 'adamw_wd': 1e-2},
        7,
    ),
]

STABLE_ADAMW_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adopt', {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 5),
    ('ademamix', {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 10),
]


class TestMaximize:
    @pytest.mark.parametrize(
        ('optimizer_name', 'foreach'),
        [
            (name, foreach)
            for name in sorted(MAXIMIZE_OPTIMIZERS)
            if name not in SKIP_CAPABILITY_PROBE
            for foreach in ([False, True] if name in FOREACH_OPTIMIZERS else [False])
        ],
    )
    def test_maximize(self, optimizer_name, foreach):
        params = [torch.full((2, 2), 2.0, requires_grad=True) for _ in range(2)]
        params[0].grad = torch.full_like(params[0], 0.5)
        params[1].grad = torch.full_like(params[1], -0.5)

        options = GRADIENT_OPTIONS.get(optimizer_name, {}).copy()
        if optimizer_name in FOREACH_OPTIMIZERS:
            options['foreach'] = foreach
        if optimizer_name == 'kron':
            options.update(pre_conditioner_update_probability=1.0, balance_prob=0.0)
        ascent = build_optimizer(optimizer_name, [params[0]], maximize=True, **options)
        descent = build_optimizer(optimizer_name, [params[1]], maximize=False, **options)

        if optimizer_name.startswith('schedulefree'):
            ascent.train()
            descent.train()

        for optimizer in (ascent, descent):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                options = {'hessian': [torch.ones_like(optimizer.param_groups[0]['params'][0])]} if (
                    optimizer_name in ('adahessian', 'sophiah')
                ) else {}
                optimizer.step(lambda: 0.1, **options)

        torch.testing.assert_close(params[0], params[1])


class TestAdaNorm:
    @pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adanorm_optimizer(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config, adanorm=True)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.75)

    @pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adanorm_variant(self, optimizer_config):
        param = make_parameter()
        param.grad = torch.ones(1, 1)

        optimizer_name, _ = optimizer_config[:2]

        optimizer = build_optimizer(optimizer_name, [param], adanorm=True)
        optimizer.step()

        param.grad = torch.zeros(1, 1)
        optimizer.step()


class TestAdamDebias:
    @pytest.mark.parametrize('optimizer_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adamd_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config, adam_debias=True)

        create_graph = optimizer_name in ('adahessian',)
        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, create_graph=create_graph, threshold=2.0)


class TestCautious:
    @pytest.mark.parametrize('optimizer_config', COPT_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_cautious_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        parameters, config = build_optimizer_parameters(model.parameters(), optimizer_name, config)
        optimizer = build_optimizer(optimizer_name, parameters, **config, cautious=True)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.5)


class TestStableAdamW:
    @pytest.mark.parametrize('optimizer_config', STABLE_ADAMW_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_stable_adamw_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.5)
