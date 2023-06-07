import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError, NoSparseGradientError
from pytorch_optimizer.base.types import BETAS, HUTCHINSON_G


class BaseOptimizer(ABC):
    r"""Base optimizer class."""

    @torch.no_grad()
    def set_hessian(self, param_groups, state, hessian):
        r"""Set hessian to state from external source. Generally useful when using functorch as a base.

        Example usage:
        ```
        # Hutchinsons Estimator using HVP
        noise = tree_map(lambda v: torch.randn_like(v), params)
        loss_, hvp_est = jvp(grad(run_model_fn), (params,), (noise,))
        hessian_diag_est  = tree_map(lambda a, b: a * b, hvp_est, noise)

        optimizer.set_hessian(hessian_diag_est)
        # OR
        optimizer.step(hessian=hessian_diag_est)
        ````
        """
        i: int = 0
        for group in param_groups:
            for p in group['params']:
                if p.shape != hessian[i].shape:
                    raise ValueError(
                        f'[-] the shape of parameter and hessian does not match. {p.size()} vs {hessian.size()}'
                    )

                state[p]['hessian'] = hessian[i]
                i += 1

    @torch.no_grad()
    def compute_hutchinson_hessian(
        self,
        param_groups,
        state,
        num_samples: int = 1,
        pre_zero: bool = True,
        alpha: float = 1.0,
        distribution: HUTCHINSON_G = 'gaussian',
    ):
        r"""Hutchinsons approximate hessian, added to the state under key `hessian`."""
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'[-] Hessian with distribution {distribution} is not implemented.')

        params = []
        for group in param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    if p.grad.is_sparse:
                        raise NoSparseGradientError(str(self))

                    if 'hessian' not in state[p]:
                        state[p]['hessian'] = torch.zeros_like(p)
                    elif pre_zero:
                        state[p]['hessian'].zero_()

                    params.append(p)

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
                state[p]['hessian'].add_(h_z * z, alpha=(1 / num_samples) * alpha)

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
    def validate_non_negative(x: float, name: str):
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
    def validate_options(x: str, name: str, options: List[str]):
        if x not in options:
            opts: str = ' or '.join([f'\'{option}\'' for option in options]).strip()
            raise ValueError(f'[-] {name} {x} must be one of ({opts})')

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]):
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    def validate_betas(self, betas: BETAS):
        self.validate_range(betas[0], 'beta1', 0.0, 1.0, range_type='[]')
        self.validate_range(betas[1], 'beta2', 0.0, 1.0, range_type='[]')

        if len(betas) < 3:
            return

        self.validate_range(betas[2], 'beta3', 0.0, 1.0, range_type='[]')

    def validate_nus(self, nus: Union[float, Tuple[float, float]]):
        if isinstance(nus, float):
            self.validate_range(nus, 'nu', 0.0, 1.0, range_type='[]')
        else:
            self.validate_range(nus[0], 'nu1', 0.0, 1.0, range_type='[]')
            self.validate_range(nus[1], 'nu2', 0.0, 1.0, range_type='[]')

    @abstractmethod
    def reset(self):
        raise NotImplementedError
