import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.type import (
    HUTCHINSON_G,
    OPTIMIZER_INSTANCE_OR_CLASS,
    Betas,
    Closure,
    Defaults,
    Loss,
    Parameters,
    ParamGroup,
    State,
)


class BaseOptimizer(ABC, Optimizer):
    """Base optimizer class. Provides common functionalities for the optimizers."""

    def __init__(self, params: Parameters, defaults: Defaults) -> None:
        super().__init__(params, defaults)

    @staticmethod
    def load_optimizer(optimizer: OPTIMIZER_INSTANCE_OR_CLASS, **kwargs) -> Optimizer:
        """Build torch.optim.Optimizer class."""
        if isinstance(optimizer, Optimizer):
            return optimizer

        if 'params' in kwargs:
            params = kwargs.pop('params')
            return optimizer(params, **kwargs)

        raise ValueError('need to pass `params` when you pass the `torch.optim.Optimizer` instance.')

    @staticmethod
    @torch.no_grad()
    def set_hessian(param_groups: Parameters, state: State, hessian: List[torch.Tensor]) -> None:
        """Set hessian to state from external source. Generally useful when using functorch as a base.

        Args:
            param_groups: PARAMETERS. Parameter groups from optimizer.
            state: STATE. Optimizer state dictionary.
            hessian: List[torch.Tensor]. Sequence of Hessian tensors to set.

        Example:
            # Hutchinson's Estimator using Hessian-vector product (HVP)
            >>> noise = tree_map(lambda v: torch.randn_like(v), params)
            >>> loss_, hvp_est = jvp(grad(run_model_fn), (params,), (noise,))
            >>> hessian_diag_est = tree_map(lambda a, b: a * b, hvp_est, noise)

            >>> optimizer.set_hessian(hessian_diag_est)
            # OR
            >>> optimizer.step(hessian=hessian_diag_est)
        """
        i: int = 0
        for group in param_groups or []:
            for p in group['params']:
                if p.size() != hessian[i].size():
                    raise ValueError(
                        f'the shape of parameter and hessian does not match. {p.size()} vs {hessian[i].size()}'
                    )

                state[p]['hessian'] = hessian[i]
                i += 1

    @staticmethod
    def zero_hessian(param_groups: Parameters, state: State, pre_zero: bool = True) -> None:
        """Zero-out Hessian.

        Args:
            param_groups (Parameters): Parameter groups from the optimizer.
            state (State): Optimizer state dictionary.
            pre_zero (bool): If True, zero-out the Hessian before computing/updating it.
        """
        for group in param_groups or []:
            for p in group['params']:
                if p.requires_grad and p.grad is not None and not p.grad.is_sparse:
                    if 'hessian' not in state[p]:
                        state[p]['hessian'] = torch.zeros_like(p)
                    elif pre_zero:
                        state[p]['hessian'].zero_()

    @staticmethod
    @torch.no_grad()
    def compute_hutchinson_hessian(
        param_groups: Parameters,
        state: State,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: HUTCHINSON_G = 'gaussian',
    ) -> None:
        r"""Hutchinson's approximate Hessian, added to the state under key `hessian`.

        Args:
            param_groups (Parameters): Parameter groups from the optimizer.
            state (State): Optimizer state dictionary.
            num_samples (int): Number of times to sample noise vector `z` for the trace approximation.
            alpha (float): Scaling factor for the Hessian estimate.
            distribution (HUTCHINSON_G): Type of noise distribution used (e.g., Rademacher).
        """
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params: List[torch.Tensor] = [
            p
            for group in param_groups or []
            for p in group['params']
            if p.requires_grad and p.grad is not None and not p.grad.is_sparse
        ]
        if len(params) == 0:
            return

        grads = [p.grad for p in params]

        for i in range(num_samples):
            if distribution == 'rademacher':
                zs = [torch.randint_like(p, 0, 1) * 2.0 - 1.0 for p in params]
            else:
                zs = [torch.randn_like(p) for p in params]

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, retain_graph=i < num_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                state[p]['hessian'].add_(h_z * z, alpha=alpha / num_samples)

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: Optional[torch.Tensor],
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: Optional[float] = None,
    ) -> None:
        """Apply weight decay in an in-place manner.

        Args:
            p (torch.Tensor): Parameter tensor to apply weight decay to.
            grad (torch.Tensor): Gradient tensor of parameter p.
            lr (float): Learning rate to scale the update.
            weight_decay (float): Weight decay coefficient (L2 penalty).
            weight_decouple (bool): If True, applies decoupled weight decay as in AdamW.
            fixed_decay (bool): If True, fixes weight decay to not depend on learning rate.
            ratio (Optional[float]): Optional scaling factor for weight decay.
        """
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def apply_cautious_weight_decay(
        p: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Apply cautious weight decay (CWD) in an in-place manner.

        Args:
            p (torch.Tensor): Parameter tensor to apply weight decay to.
            update (torch.Tensor): update tensor.
            lr (float): Learning rate to scale the update.
            weight_decay (float): Weight decay coefficient (L2 penalty).
        """
        p.copy_(torch.where(update * p >= 0, p * (1.0 - weight_decay * lr), p))

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        eps: float,
        exp_avg_sq_eps: float = 1e-15,
    ) -> torch.Tensor:
        """Apply AMSBound variant.

        Args:
            ams_bound (bool): Whether to apply the AMSBound variant.
            exp_avg_sq (torch.Tensor): Exponential moving average of squared gradients.
            max_exp_avg_sq (Optional[torch.Tensor]): Maximum of all exp_avg_sq elements, for AMSBound.
            eps (float): Small epsilon value for numerical stability.
            exp_avg_sq_eps (float): Epsilon used specifically for numerical stability in exp_avg_sq computations.
        """
        if ams_bound:
            if torch.is_complex(max_exp_avg_sq):
                max_exp_avg_sq = torch.view_as_real(max_exp_avg_sq)

            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            de_nom = max_exp_avg_sq.add(exp_avg_sq_eps)
        else:
            de_nom = exp_avg_sq.add(exp_avg_sq_eps)

        return de_nom.sqrt_().add_(eps)

    @staticmethod
    def debias(beta: float, step: int) -> float:
        """Adam-style debias correction.

        Args:
            beta (float): Exponential decay rate for moment estimates.
            step (int): Current optimization step number.
        """
        return 1.0 - math.pow(beta, step)  # fmt: skip

    @staticmethod
    def debias_beta(beta: float, step: int) -> float:
        r"""Apply the Adam-style debias correction into beta.

        Simplified version of `\^{beta} = beta * (1.0 - beta ** (step - 1)) / (1.0 - beta ** step)`

        Args:
            beta (float): The original beta decay rate.
            step (int): Current optimization step number.
        """
        beta_n: float = math.pow(beta, step)
        return (beta_n - beta) / (beta_n - 1.0)  # fmt: skip

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        """Apply AdamD variant.

        Args:
            adam_debias (bool): If True, only corrects the denominator to avoid inflating step sizes early in training.
            step_size (float): The step size for the update.
            bias_correction1 (float): The bias correction factor for the first moment.
        """
        return step_size if adam_debias else step_size / bias_correction1

    @staticmethod
    def get_rectify_step_size(
        is_rectify: bool,
        step: int,
        lr: float,
        beta2: float,
        n_sma_threshold: int,
        degenerated_to_sgd: bool,
    ) -> Tuple[float, float]:
        """Get step size for rectify optimizer.

        Args:
            is_rectify (bool): Whether to apply the rectify variant.
            step (int): Current step number.
            lr (float): Base learning rate.
            beta2 (float): Beta2 parameter from optimizer (momentum term).
            n_sma_threshold (float): Simple Moving Average (SMA) threshold for rectification.
            degenerated_to_sgd (bool): Whether to degenerate to SGD if below threshold.
        """
        step_size: float = lr
        n_sma: float = 0.0

        if is_rectify:
            n_sma_max: float = 2.0 / (1.0 - beta2) - 1.0
            beta2_t: float = beta2 ** step  # fmt: skip
            n_sma: float = n_sma_max - 2 * step * beta2_t / (1.0 - beta2_t)

            if n_sma >= n_sma_threshold:
                rt = math.sqrt(
                    (1.0 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2)
                )
            elif degenerated_to_sgd:
                rt = 1.0
            else:
                rt = -1.0

            step_size *= rt

        return step_size, n_sma

    @staticmethod
    def get_adanorm_gradient(
        grad: torch.Tensor, adanorm: bool, exp_grad_norm: Optional[torch.Tensor] = None, r: Optional[float] = 0.95
    ) -> torch.Tensor:
        r"""Get AdaNorm gradient.

        Args:
            grad (torch.Tensor): Gradient.
            adanorm (bool): Whether to use the AdaNorm variant.
            exp_grad_norm (Optional[torch.Tensor]): Exponential moving average of gradient norm.
            r (Optional[float]): EMA factor; between 0.9 and 0.99 is preferred.
        """
        if not adanorm or exp_grad_norm is None:
            return grad

        if r is None:
            r = 0.95

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul(r).add_(grad_norm, alpha=1.0 - r)

        return grad.mul(exp_grad_norm).div_(grad_norm) if exp_grad_norm > grad_norm else grad

    @staticmethod
    def get_rms(x: torch.Tensor) -> torch.Tensor:
        """Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Get approximation of EMA of squared gradient."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    @staticmethod
    def apply_cautious(update: torch.Tensor, grad: torch.Tensor) -> None:
        """Apply the Cautious Optimizer feature.

        Args:
            update (torch.Tensor): update. it'll be masked in in-place manner.
            grad (torch.Tensor): gradient.
        """
        mask = (update * grad > 0).to(grad.dtype)
        mask.mul_(mask.numel() / (mask.sum() + 1))
        update.mul_(mask)

    @staticmethod
    def get_stable_adamw_rms(grad: torch.Tensor, exp_avg_sq: torch.Tensor, eps: float = 1e-16) -> float:
        """Get StableAdamW RMS.

        Args:
            grad (torch.Tensor): gradient.
            exp_avg_sq (torch.Tensor): Exponential moving average of squared gradient.
            eps (float): Small value to prevent division by zero.
        """
        return grad.pow(2).div_(exp_avg_sq.clip(min=eps)).mean().sqrt_().clip_(min=1.0).item()

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)') -> None:
        if range_type == '[)' and not low <= x < high:
            raise ValueError(f'{name} must be in the range [{low}, {high})')
        if range_type == '[]' and not low <= x <= high:
            raise ValueError(f'{name} must be in the range [{low}, {high}]')
        if range_type == '(]' and not low < x <= high:
            raise ValueError(f'{name} must be in the range ({low}, {high}]')
        if range_type == '()' and not low < x < high:
            raise ValueError(f'{name} must be in the range ({low}, {high})')

    @staticmethod
    def validate_non_negative(x: Optional[float], name: str) -> None:
        if x is not None and x < 0.0:
            raise ValueError(f'{name} must be non-negative')

    @staticmethod
    def validate_non_positive(x: Optional[float], name: str) -> None:
        if x is not None and x > 0.0:
            raise ValueError(f'{name} must be non-positive')

    @staticmethod
    def validate_positive(x: Union[float, int], name: str) -> None:
        if x <= 0:
            raise ValueError(f'{name} must be positive')

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper') -> None:
        if bound_type == 'upper' and constant > boundary:
            raise ValueError(f'constant {constant} must be in a range of (-inf, {boundary}]')
        if bound_type == 'lower' and constant < boundary:
            raise ValueError(f'constant {constant} must be in a range of [{boundary}, inf)')

    @staticmethod
    def validate_step(step: int, step_type: str) -> None:
        if step < 1:
            raise NegativeStepError(step, step_type=step_type)

    @staticmethod
    def validate_options(x: str, name: str, options: List[str]) -> None:
        if x not in options:
            opts: str = ' or '.join([f"'{option}'" for option in options]).strip()
            raise ValueError(f'{name} {x} must be one of ({opts})')

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]) -> None:
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_mod(x: int, y: int) -> None:
        if x % y != 0:
            raise ValueError(f'{x} must be divisible by {y}')

    def validate_betas(self, betas: Betas, beta_range_type: str = '[)', beta3_range_type: str = '[]') -> None:
        if betas[0] is not None:
            self.validate_range(betas[0], 'beta1', 0.0, 1.0, range_type=beta_range_type)

        self.validate_range(betas[1], 'beta2', 0.0, 1.0, range_type=beta_range_type)

        if len(betas) < 3:
            return

        if betas[2] is not None:
            self.validate_range(betas[2], 'beta3', 0.0, 1.0, range_type=beta3_range_type)

    def validate_nus(self, nus: Union[float, Tuple[float, float]]) -> None:
        if isinstance(nus, tuple):
            nu1, nu2 = nus
            self.validate_range(nu1, 'nu1', 0.0, 1.0, range_type='[]')
            self.validate_range(nu2, 'nu2', 0.0, 1.0, range_type='[]')
        else:
            self.validate_range(nus, 'nu', 0.0, 1.0, range_type='[]')

    @abstractmethod
    def init_group(self, group: ParamGroup, **kwargs) -> None:  # pragma: no cover
        """Initialize the group of the optimizer and return is_complex."""
        return

    @staticmethod
    def view_as_real(param, *state_and_grads) -> tuple:
        """View imaginary tensors as real tensors."""
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            state_and_grads = tuple(
                torch.view_as_real(s) if (s is not None and torch.is_complex(s)) else s if s is not None else None
                for s in state_and_grads
            )

        return param, *state_and_grads

    @staticmethod
    def maximize_gradient(grad: torch.Tensor, maximize: bool = False) -> None:
        """Maximize the objective with respect to the params, instead of minimizing."""
        if maximize:
            grad.neg_()

    def step(self, closure: Closure = None) -> Loss:  # pragma: no cover
        raise NotImplementedError
