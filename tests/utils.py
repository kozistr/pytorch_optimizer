from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
from torch.optim import AdamW

from pytorch_optimizer.base.type import Loss
from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad, ScheduleFreeWrapper, load_optimizer
from pytorch_optimizer.optimizer.alig import l2_projection


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = relu(x)
        return self.fc2(x)


class ComplexLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, dtype=torch.complex64)
        self.fc2 = nn.Linear(2, 1, dtype=torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = relu(x.real) + 1.0j * relu(x.imag)
        return self.fc2(x).real


class MultiHeadLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.head1 = nn.Linear(2, 1)
        self.head2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = relu(x)
        return self.head1(x), self.head2(x)


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.fc1(x))


class MultiClassExample(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def simple_zero_rank_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.tensor(0.0).requires_grad_(require_grad)
    param.grad = torch.tensor(0.0)
    return param


def simple_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.zeros(1, 1).requires_grad_(require_grad)
    param.grad = torch.zeros(1, 1)
    return param


def simple_complex_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.zeros(1, 1, dtype=torch.complex64).requires_grad_(require_grad)
    param.grad = torch.randn(1, 1, dtype=torch.complex64)
    return param


def simple_sparse_parameter(require_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(5, 1).requires_grad_(require_grad)
    weight_sparse = weight.detach().requires_grad_(require_grad)

    if require_grad:
        weight.grad = torch.rand_like(weight)
        weight.grad[0] = 0.0
        weight_sparse.grad = weight.grad.to_sparse()

    return weight, weight_sparse


def dummy_closure() -> Loss:
    return 1.0


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamW(*parameters, **kwargs))


def build_orthograd(*parameters, **kwargs):
    return OrthoGrad(AdamW(*parameters, **kwargs))


def build_schedulefree(*parameters, **kwargs):
    return ScheduleFreeWrapper(AdamW(*parameters, **kwargs))


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'


def names(v) -> str:
    return v.__name__


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sphere_loss(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum()  # fmt: skip


def build_model(use_complex: bool = False):
    torch.manual_seed(42)
    return ComplexLogisticRegression() if use_complex else LogisticRegression(), nn.BCEWithLogitsLoss()


def build_optimizer_parameter(parameters, optimizer_name, config):
    if optimizer_name == 'AliG':
        config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})
    elif optimizer_name in ('Muon', 'AdaMuon', 'AdaGO'):
        hidden_weights = [p for p in parameters if p.ndim >= 2]
        hidden_gains_biases = [p for p in parameters if p.ndim < 2]

        parameters = [
            {'params': hidden_weights, 'use_muon': True},
            {'params': hidden_gains_biases, 'use_muon': False},
        ]
    elif optimizer_name == 'AdamWSN':
        sn_params = [p for p in parameters if p.ndim == 2]
        regular_params = [p for p in parameters if p.ndim != 2]
        parameters = [{'params': sn_params, 'sn': True}, {'params': regular_params, 'sn': False}]
    elif optimizer_name == 'AdamC':
        norm_params = [p for i, p in enumerate(parameters) if i == 1]
        regular_params = [p for i, p in enumerate(parameters) if i != 1]
        parameters = [{'params': norm_params, 'normalized': True}, {'params': regular_params}]

    return parameters, config


def make_closure(value):
    """Create a closure that returns the given value."""

    def closure():
        return value

    return closure


def should_use_create_graph(optimizer_name: str) -> bool:
    """Check if optimizer requires create_graph=True for backward."""
    return optimizer_name.lower() in ('adahessian', 'sophiah')


class OptimizerBuilder:
    """Builder class for creating optimizers with special configurations."""

    @staticmethod
    def with_muon(params, use_muon: bool):
        """Wrap parameters with use_muon flag for Muon-based optimizers."""

        def with_flag(group):
            if isinstance(group, dict):
                return group if 'use_muon' in group else {**group, 'use_muon': use_muon}
            return {'params': group, 'use_muon': use_muon}

        return [with_flag(group) for group in params] if isinstance(params, list) else [with_flag(params)]

    @classmethod
    def create(cls, name: str, params: List, **overrides):
        """Create an optimizer by name with optional overrides."""
        optimizer_name: str = name.lower()

        if optimizer_name == 'lookahead':
            return Lookahead(load_optimizer('adamw')(params), k=1)
        if optimizer_name == 'trac':
            return TRAC(load_optimizer('adamw')(params))
        if optimizer_name == 'orthograd':
            return OrthoGrad(load_optimizer('adamw')(params))

        if optimizer_name == 'ranger21':
            overrides.update({'num_iterations': 1, 'lookahead_merge_time': 1})
        elif optimizer_name == 'bsam':
            overrides.update({'num_data': 1})
        elif optimizer_name in ('lamb', 'ralamb'):
            overrides.update({'pre_norm': True})
        elif optimizer_name == 'alice':
            overrides.update({'rank': 2, 'leading_basis': 1})
        elif optimizer_name == 'adahessian':
            overrides.update({'update_period': 2})

        if optimizer_name in ('muon', 'adamuon', 'adago'):
            params = cls.with_muon(params, use_muon=overrides.pop('use_muon', False))

        return load_optimizer(optimizer_name)(params, **overrides)


class TrainingRunner:
    """A utility class for running training loops in tests.

    This class encapsulates the common training loop pattern used across
    multiple test files to reduce code duplication.
    """

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

    def run(
        self,
        iterations: int = 5,
        create_graph: bool = False,
        closure_fn=None,
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run standard training loop and assert loss reduction.

        Args:
            iterations: Number of training iterations.
            create_graph: Whether to create graph during backward.
            closure_fn: Optional closure function for optimizers that require it.
            threshold: Loss reduction threshold (init_loss > threshold * final_loss).

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """
        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            self.optimizer.zero_grad()

            y_pred = self.model(self.x_data)
            loss = self.loss_fn(y_pred, self.y_data)

            if init_loss == np.inf:
                init_loss = loss

            loss.backward(create_graph=create_graph)

            if closure_fn is not None:
                self.optimizer.step(closure_fn(loss))
            else:
                self.optimizer.step()

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_bf16(
        self,
        iterations: int = 5,
        create_graph: bool = False,
        closure_fn=None,
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run bf16 training loop with autocast.

        Args:
            iterations: Number of training iterations.
            create_graph: Whether to create graph during backward.
            closure_fn: Optional closure function for optimizers that require it.
            threshold: Loss reduction threshold (init_loss > threshold * final_loss).

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """
        context = torch.autocast('cpu', dtype=torch.bfloat16)
        scaler = torch.GradScaler(device='cpu', enabled=False)

        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            self.optimizer.zero_grad()

            with context:
                loss = self.loss_fn(self.model(self.x_data), self.y_data)

            if init_loss == np.inf:
                init_loss = loss

            scaler.scale(loss).backward(create_graph=create_graph)

            if closure_fn is not None:
                self.optimizer.step(closure_fn(loss))
            else:
                self.optimizer.step()

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_sam_style(
        self,
        iterations: int = 3,
        threshold: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SAM-style two-step training loop.

        Args:
            iterations: Number of training iterations.
            threshold: Loss reduction threshold.

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """
        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            loss = self.loss_fn(self.y_data, self.model(self.x_data))
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.loss_fn(self.y_data, self.model(self.x_data)).backward()
            self.optimizer.second_step(zero_grad=True)

            if init_loss == np.inf:
                init_loss = loss

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_with_closure(
        self,
        iterations: int = 3,
        threshold: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run training loop with closure-based step.

        Args:
            iterations: Number of training iterations.
            threshold: Loss reduction threshold.

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """

        def closure():
            first_loss = self.loss_fn(self.y_data, self.model(self.x_data))
            first_loss.backward()
            return first_loss

        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            loss = self.loss_fn(self.y_data, self.model(self.x_data))
            loss.backward()

            self.optimizer.step(closure)
            self.optimizer.zero_grad()

            if init_loss == np.inf:
                init_loss = loss

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_wsam_style(
        self,
        iterations: int = 10,
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run WSAM-style training loop.

        Args:
            iterations: Number of training iterations.
            threshold: Loss reduction threshold.

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """
        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            loss = self.loss_fn(self.y_data, self.model(self.x_data))
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.loss_fn(self.y_data, self.model(self.x_data)).backward()
            self.optimizer.second_step(zero_grad=True)

            if init_loss == np.inf:
                init_loss = loss

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_wsam_with_closure(
        self,
        iterations: int = 10,
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run WSAM with closure-based training loop.

        Args:
            iterations: Number of training iterations.
            threshold: Loss reduction threshold.

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """

        def closure():
            output = self.model(self.x_data)
            _loss = self.loss_fn(output, self.y_data)
            _loss.backward()
            return _loss

        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            loss = self.optimizer.step(closure)
            self.optimizer.zero_grad()

            if init_loss == np.inf:
                init_loss = loss

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np

    def run_trac_style(
        self,
        iterations: int = 3,
        threshold: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run TRAC-style training loop.

        Args:
            iterations: Number of training iterations.
            threshold: Loss reduction threshold.

        Returns:
            Tuple of (init_loss, final_loss) as numpy arrays.
        """
        init_loss, loss = np.inf, np.inf
        for _ in range(iterations):
            loss = self.loss_fn(self.model(self.x_data), self.y_data)

            if init_loss == np.inf:
                init_loss = loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        init_loss_np = tensor_to_numpy(init_loss)
        final_loss_np = tensor_to_numpy(loss)

        assert (
            init_loss_np > threshold * final_loss_np
        ), f'Loss did not decrease enough: {init_loss_np:.4f} > {threshold} * {final_loss_np:.4f}'

        return init_loss_np, final_loss_np


class LRSchedulerAssertions:
    """Utility class for LR scheduler testing assertions."""

    @staticmethod
    def assert_lr_sequence(scheduler, expected_lrs, decimals: int = 7) -> None:
        """Assert that the scheduler produces the expected learning rate sequence."""
        for expected_lr in expected_lrs:
            scheduler.step()
            np.testing.assert_almost_equal(expected_lr, scheduler.get_lr(), decimals)
