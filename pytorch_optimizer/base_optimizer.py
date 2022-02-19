from abc import ABC, abstractmethod

from pytorch_optimizer.types import BETAS


class BaseOptimizer(ABC):
    @staticmethod
    def validate_learning_rate(learning_rate: float):
        if learning_rate < 0.0:
            raise ValueError(f'[-] learning rate {learning_rate} must be positive')

    @staticmethod
    def validate_betas(betas: BETAS):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'[-] beta0 {betas[0]} must be in the range [0, 1)')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'[-] beta1 {betas[1]} must be in the range [0, 1)')

    @staticmethod
    def validate_weight_decay(weight_decay: float):
        if weight_decay < 0.0:
            raise ValueError(f'[-] weight_decay {weight_decay} must be non-negative')

    @staticmethod
    def validate_weight_decay_ratio(weight_decay_ratio: float):
        if not 0.0 <= weight_decay_ratio < 1.0:
            raise ValueError(f'[-] weight_decay_ratio {weight_decay_ratio} must be in the range [0, 1)')

    @staticmethod
    def validate_hessian_power(hessian_power: float):
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f'[-] hessian_power {hessian_power} must be in the range [0, 1]')

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
        if k < 0:
            raise ValueError(f'[-] k {k} must be non-negative')

    @staticmethod
    def validate_epsilon(epsilon: float):
        if epsilon < 0.0:
            raise ValueError(f'[-] epsilon {epsilon} must be non-negative')

    @abstractmethod
    def validate_parameters(self):
        raise NotImplementedError
