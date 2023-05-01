import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.types import BETAS


class BaseOptimizer(ABC):
    r"""Base optimizer class."""

    @staticmethod
    def get_rectify_step_size(
        is_rectify: bool,
        step: int,
        lr: float,
        beta2: float,
        bias_correction1: float,
        n_sma_threshold: int,
        degenerated_to_sgd: bool,
        adam_debias: bool,
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

        if not adam_debias:
            step_size /= bias_correction1

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
    def validate_learning_rate(learning_rate: float):
        if learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_beta(beta: float):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f'[-] beta {beta} must be in the range [0, 1]')

    @staticmethod
    def validate_beta0(beta_0: float):
        if not 0.0 <= beta_0 <= 1.0:
            raise ValueError(f'[-] beta0 {beta_0} must be in the range [0, 1]')

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
    def validate_weight_decay(weight_decay: float):
        if weight_decay < 0.0:
            raise ValueError(f'[-] weight_decay {weight_decay} must be non-negative')

    @staticmethod
    def validate_weight_decay_type(weight_decay_type: str):
        if weight_decay_type not in ('l2', 'decoupled', 'stable'):
            raise ValueError(
                f'[-] weight_decay_type {weight_decay_type} must be one of (\'l2\', \'decoupled\', \'stable\')'
            )

    @staticmethod
    def validate_weight_decay_ratio(weight_decay_ratio: float):
        if not 0.0 <= weight_decay_ratio < 1.0:
            raise ValueError(f'[-] weight_decay_ratio {weight_decay_ratio} must be in the range [0, 1)')

    @staticmethod
    def validate_trust_coefficient(trust_coefficient: float):
        if trust_coefficient < 0.0:
            raise ValueError(f'[-] trust_coefficient {trust_coefficient} must be non-negative')

    @staticmethod
    def validate_momentum(momentum: float):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'[-] momentum {momentum} must be in the range [0, 1)')

    @staticmethod
    def validate_lookahead_k(k: int):
        if k < 1:
            raise ValueError(f'[-] k {k} must be positive')

    @staticmethod
    def validate_rho(rho: float):
        if rho < 0.0:
            raise ValueError(f'[-] rho {rho} must be non-negative')

    @staticmethod
    def validate_epsilon(epsilon: float):
        if epsilon < 0.0:
            raise ValueError(f'[-] epsilon {epsilon} must be non-negative')

    @staticmethod
    def validate_alpha(alpha: float):
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f'[-] alpha {alpha} must be in the range [0, 1)')

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
    def validate_update_frequency(update_frequency: int):
        if update_frequency < 1:
            raise NegativeStepError(update_frequency, step_type='update_frequency')

    @staticmethod
    def validate_norm(norm: float):
        if norm < 0.0:
            raise ValueError(f'[-] norm {norm} must be positive')

    @staticmethod
    def validate_rebound(rebound: str):
        if rebound not in ('constant', 'belief'):
            raise ValueError(f'[-] rebound {rebound} must be one of (\'constant\' or \'belief\')')

    @staticmethod
    def validate_lipschitz_constant(lips: float):
        if lips < 0:
            raise ValueError(f'[-] Lipschitz constant {lips} must be non-negative')

    @staticmethod
    def validate_a2grad_variant(variant: str):
        if variant not in ('uni', 'inc', 'exp'):
            raise ValueError(f'[-] A2Grad variant {variant} must be one of (\'uni\' or \'inc\' or \'exp\')')

    @staticmethod
    def validate_kappa(kappa: float):
        if kappa < 0.0:
            raise ValueError(f'[-] kappa {kappa} must be non-negative')

    @staticmethod
    def validate_xi(xi: float):
        if xi < 0.0:
            raise ValueError(f'[-] xi {xi} must be non-negative')

    @staticmethod
    def validate_constant(constant: float, boundary: float):
        if constant > boundary:
            raise ValueError(f'[-] constant {constant} must be in a range of (-inf, {boundary}]')

    @staticmethod
    def validate_amplifier(amplifier: float):
        if amplifier < 0.0:
            raise ValueError(f'[-] amplifier {amplifier} must be non-negative')

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
    def validate_parameters(self):
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def reset(self):
        raise NotImplementedError
