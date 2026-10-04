from collections.abc import Iterable
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pytorch_optimizer.base.type import Loss, ParamsT
from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad, ScheduleFreeWrapper, load_optimizer
from pytorch_optimizer.optimizer.alig import l2_projection
from pytorch_optimizer.optimizer.utils import HAS_TRANSFORMERS


def build_optimizer(name: str, params: ParamsT | nn.Module, **options) -> Optimizer:
    name = name.lower()
    wrappers = {'lookahead': Lookahead, 'orthograd': OrthoGrad, 'schedulefree': ScheduleFreeWrapper, 'trac': TRAC}
    if name in wrappers:
        wrapper_options = {'k': options.pop('k')} if name == 'lookahead' and 'k' in options else {}
        return wrappers[name](load_optimizer('adamw')(params, **options), **wrapper_options)

    defaults = {
        'ranger21': {'num_iterations': 1},
        'bsam': {'num_data': 1},
        'sgd': {'lr': 1e-3},
    }
    options = {**defaults.get(name, {}), **options}
    use_muon = options.pop('use_muon', False)
    if name in ('muon', 'adamuon', 'adago', 'normuon'):
        params = list(params) if isinstance(params, Iterable) else params
        params = [
            {**group, 'use_muon': group.get('use_muon', use_muon)}
            if isinstance(group, dict)
            else {'params': [group], 'use_muon': use_muon}
            for group in params
        ]
    warning = (
        pytest.warns(ImportWarning, match='you need to install `transformers`')
        if name == 'adalomo' and not HAS_TRANSFORMERS
        else nullcontext()
    )
    with warning:
        return load_optimizer(name)(params, **options)


def dummy_closure() -> Loss:
    return 1.0


def ids(v) -> str:
    return f'{v[0]}_{v[1:]}'


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sphere_loss(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum()


def build_optimizer_parameters(parameters, optimizer_name, config):
    config = config.copy()
    if isinstance(parameters, nn.Module):
        return parameters, config

    parameters = list(parameters)
    if optimizer_name == 'alig':
        config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})
    elif optimizer_name in ('muon', 'adamuon', 'adago', 'normuon'):
        hidden_weights = [p for p in parameters if p.ndim >= 2]
        hidden_gains_biases = [p for p in parameters if p.ndim < 2]

        parameters = [
            {'params': hidden_weights, 'use_muon': True},
            {'params': hidden_gains_biases, 'use_muon': False},
        ]
    elif optimizer_name in ('spectralsphere',):
        parameters = [{'params': [p for p in parameters if p.ndim >= 2]}]
    elif optimizer_name == 'adamwsn':
        sn_params = [p for p in parameters if p.ndim == 2]
        regular_params = [p for p in parameters if p.ndim != 2]
        parameters = [{'params': sn_params, 'sn': True}, {'params': regular_params, 'sn': False}]
    elif optimizer_name == 'adamc':
        norm_params = [p for i, p in enumerate(parameters) if i == 1]
        regular_params = [p for i, p in enumerate(parameters) if i != 1]
        parameters = [{'params': norm_params, 'normalized': True}, {'params': regular_params}]

    return parameters, config


def make_closure(value):
    def closure():
        return value

    return closure


def should_use_create_graph(optimizer_name: str) -> bool:
    return optimizer_name.lower() in ('adahessian', 'sophiah')


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.x_data = x_data
        self.y_data = y_data

    def compute_loss(self, swap_args: bool = False) -> torch.Tensor:
        y_pred = self.model(self.x_data)
        if swap_args:
            return self.loss_fn(self.y_data, y_pred)
        return self.loss_fn(y_pred, self.y_data)

    def assert_loss_decreased(
        self,
        init_loss: torch.Tensor,
        final_loss: torch.Tensor,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        for p in self.model.parameters():
            assert torch.isfinite(p).all(), 'Model parameters became nonfinite during training'

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(final_loss)

        assert init_loss_np > threshold * final_loss_np, (
            f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'
        )

        return init_loss_np, final_loss_np

    def run(
        self,
        iterations: int = 5,
        create_graph: bool = False,
        closure_fn=None,
        threshold: float = 1.5,
        use_amp: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        init_loss, loss = None, None
        for _ in range(iterations):
            self.optimizer.zero_grad()

            with torch.autocast(self.x_data.device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = self.compute_loss()

            init_loss = init_loss or loss

            if create_graph:
                parameters = [p for p in self.model.parameters() if p.requires_grad]
                gradients = torch.autograd.grad(loss, parameters, create_graph=True)
                for param, grad in zip(parameters, gradients):
                    param.grad = grad
            else:
                loss.backward()

            if closure_fn is not None:
                self.optimizer.step(closure_fn(loss))
            else:
                self.optimizer.step()

        return self.assert_loss_decreased(init_loss, loss, threshold)

    def run_sam_style(self, iterations: int = 3, threshold: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        init_loss, loss = None, None
        for _ in range(iterations):
            loss = self.compute_loss(swap_args=True)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.compute_loss(swap_args=True).backward()
            self.optimizer.second_step(zero_grad=True)

            init_loss = init_loss or loss

        return self.assert_loss_decreased(init_loss, loss, threshold)

    def run_with_closure(self, iterations: int = 3, threshold: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        def closure():
            first_loss = self.compute_loss(swap_args=True)
            first_loss.backward()
            return first_loss

        init_loss, loss = None, None
        for _ in range(iterations):
            loss = self.compute_loss(swap_args=True)
            loss.backward()

            self.optimizer.step(closure)
            self.optimizer.zero_grad()

            init_loss = init_loss or loss

        return self.assert_loss_decreased(init_loss, loss, threshold)

    def run_wsam_with_closure(self, iterations: int = 10, threshold: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        def closure():
            _loss = self.compute_loss()
            _loss.backward()
            return _loss

        init_loss, loss = None, None
        for _ in range(iterations):
            loss = self.optimizer.step(closure)
            self.optimizer.zero_grad()

            init_loss = init_loss or loss

        return self.assert_loss_decreased(init_loss, loss, threshold)


class LRSchedulerAssertions:
    @staticmethod
    def assert_lr_sequence(scheduler, expected_lrs, decimals: int = 7) -> None:
        for expected_lr in expected_lrs:
            if isinstance(scheduler, LRScheduler):
                scheduler.optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr() if hasattr(scheduler, 'last_lr') else scheduler.get_last_lr()
            np.testing.assert_almost_equal(expected_lr, lr, decimals)
