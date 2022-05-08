from abc import ABC, abstractmethod

import torch

from pytorch_optimizer.types import BETAS


class BaseOptimizer(ABC):
    @staticmethod
    def validate_learning_rate(learning_rate: float):
        if learning_rate < 0.0:
            raise ValueError(f'[-] learning rate {learning_rate} must be positive')

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
            raise ValueError(f'[-] update_frequency {update_frequency} must be positive')

    @abstractmethod
    def validate_parameters(self):
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def reset(self):
        raise NotImplementedError
