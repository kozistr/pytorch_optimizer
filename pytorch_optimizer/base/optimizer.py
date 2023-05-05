import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.types import BETAS


class BaseOptimizer(ABC):
    r"""Base optimizer class."""

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: Optional[torch.Tensor],
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: Optional[float] = None,
    ):
        r"""Apply weight decay."""
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool, exp_avg_sq: torch.Tensor, max_exp_avg_sq: Optional[torch.Tensor], eps: float
    ) -> torch.Tensor:
        r"""Apply AMSBound variant."""
        if ams_bound:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            de_nom = max_exp_avg_sq.add(eps)
        else:
            de_nom = exp_avg_sq.add(eps)

        return de_nom.sqrt_().add_(eps)

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        r"""Apply AdamD variant."""
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
        r"""Get step size for rectify optimizer."""
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
        r"""Get AdaNorm gradient."""
        if not adanorm:
            return grad

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul_(r).add_(grad_norm, alpha=1.0 - r)

        return grad * exp_grad_norm / grad_norm if exp_grad_norm > grad_norm else grad

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]):
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_betas(betas: BETAS):
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError(f'[-] beta1 {betas[0]} must be in the range [0, 1]')
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f'[-] beta2 {betas[1]} must be in the range [0, 1]')

        if len(betas) < 3:
            return

        if not 0.0 <= betas[2] <= 1.0:
            raise ValueError(f'[-] beta3 {betas[2]} must be in the range [0, 1]')

    @staticmethod
    def validate_weight_decay_type(weight_decay_type: str):
        if weight_decay_type not in ('l2', 'decoupled', 'stable'):
            raise ValueError(
                f'[-] weight_decay_type {weight_decay_type} must be one of (\'l2\', \'decoupled\', \'stable\')'
            )

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)'):
        if range_type == '[)' and not low <= x < high:
            raise ValueError(f'[-] {name} must be in the range [{low}, {high})')
        if range_type == '[]' and not low <= x <= high:
            raise ValueError(f'[-] {name} must be in the range [{low}, {high}]')
        if range_type == '(]' and not low < x <= high:
            raise ValueError(f'[-] {name} must be in the range ({low}, {high}]')
        if range_type == '()' and not low < x < high:
            raise ValueError(f'[-] {name} must be in the range ({low}, {high})')

    @staticmethod
    def validate_negative(x: float, name: str):
        if x < 0.0:
            raise ValueError(f'[-] {name} must be non-negative')

    @staticmethod
    def validate_positive(x: Union[float, int], name: str):
        if x < 1:
            raise ValueError(f'[-] {name} must be positive')

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper'):
        if bound_type == 'upper' and constant > boundary:
            raise ValueError(f'[-] constant {constant} must be in a range of (-inf, {boundary}]')
        if bound_type == 'lower' and constant < boundary:
            raise ValueError(f'[-] constant {constant} must be in a range of [{boundary}, inf)')

    @staticmethod
    def validate_step(step: int, step_type: str):
        if step < 1:
            raise NegativeStepError(step, step_type=step_type)

    @staticmethod
    def validate_pullback_momentum(pullback_momentum: str):
        if pullback_momentum not in ('none', 'reset', 'pullback'):
            raise ValueError(
                f'[-] pullback_momentum {pullback_momentum} must be one of (\'none\' or \'reset\' or \'pullback\')'
            )

    @staticmethod
    def validate_reduction(reduction: str):
        if reduction not in ('mean', 'sum'):
            raise ValueError(f'[-] reduction {reduction} must be one of (\'mean\' or \'sum\')')

    @staticmethod
    def validate_rebound(rebound: str):
        if rebound not in ('constant', 'belief'):
            raise ValueError(f'[-] rebound {rebound} must be one of (\'constant\' or \'belief\')')

    @staticmethod
    def validate_a2grad_variant(variant: str):
        if variant not in ('uni', 'inc', 'exp'):
            raise ValueError(f'[-] A2Grad variant {variant} must be one of (\'uni\' or \'inc\' or \'exp\')')

    @staticmethod
    def validate_nus(nus: Union[float, Tuple[float, float]]):
        if isinstance(nus, float):
            if not 0.0 <= nus <= 1.0:
                raise ValueError(f'[-] nus {nus} must be in the range [0, 1]')
        else:
            if not 0.0 <= nus[0] <= 1.0:
                raise ValueError(f'[-] nus1 {nus[0]} must be in the range [0, 1]')
            if not 0.0 <= nus[1] <= 1.0:
                raise ValueError(f'[-] nus2 {nus[1]} must be in the range [0, 1]')

    @abstractmethod
    def reset(self):
        raise NotImplementedError
