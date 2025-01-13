import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, HUTCHINSON_G, LOSS, PARAMETERS, STATE


class BaseOptimizer(ABC, Optimizer):
    r"""Base optimizer class. Provides common functionalities for the optimizers."""

    def __init__(self, params: PARAMETERS, defaults: DEFAULTS) -> None:
        super().__init__(params, defaults)

    @staticmethod
    @torch.no_grad()
    def set_hessian(param_groups: PARAMETERS, state: STATE, hessian: List[torch.Tensor]) -> None:
        r"""Set hessian to state from external source. Generally useful when using functorch as a base.

        Example:
        -------
            Here's an example::

                # Hutchinson's Estimator using HVP
                noise = tree_map(lambda v: torch.randn_like(v), params)
                loss_, hvp_est = jvp(grad(run_model_fn), (params,), (noise,))
                hessian_diag_est  = tree_map(lambda a, b: a * b, hvp_est, noise)

                optimizer.set_hessian(hessian_diag_est)
                # OR
                optimizer.step(hessian=hessian_diag_est)

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param hessian: List[torch.Tensor]. sequence of hessian to set.
        """
        i: int = 0
        for group in param_groups:
            for p in group['params']:
                if p.size() != hessian[i].size():
                    raise ValueError(
                        f'[-] the shape of parameter and hessian does not match. {p.size()} vs {hessian[i].size()}'
                    )

                state[p]['hessian'] = hessian[i]
                i += 1

    @staticmethod
    def zero_hessian(param_groups: PARAMETERS, state: STATE, pre_zero: bool = True) -> None:
        r"""Zero-out hessian.

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param pre_zero: bool. zero-out hessian before computing the hessian.
        """
        for group in param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None and not p.grad.is_sparse:
                    if 'hessian' not in state[p]:
                        state[p]['hessian'] = torch.zeros_like(p)
                    elif pre_zero:
                        state[p]['hessian'].zero_()

    @staticmethod
    @torch.no_grad()
    def compute_hutchinson_hessian(
        param_groups: PARAMETERS,
        state: STATE,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: HUTCHINSON_G = 'gaussian',
    ) -> None:
        r"""Hutchinson's approximate hessian, added to the state under key `hessian`.

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param num_samples: int. number of times to sample `z` for the approximation of the hessian trace.
        :param alpha: float. alpha.
        :param distribution: HUTCHINSON_G. type of distribution.
        """
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'[-] Hessian with distribution {distribution} is not implemented.')

        params: List[torch.Tensor] = [
            p
            for group in param_groups
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
        r"""Apply weight decay.

        :param p: torch.Tensor. parameter.
        :param grad: torch.Tensor. gradient.
        :param lr: float. learning rate.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param fixed_decay: bool. fix weight decay.
        :param ratio: Optional[float]. scale weight decay.
        """
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool, exp_avg_sq: torch.Tensor, max_exp_avg_sq: Optional[torch.Tensor], eps: float
    ) -> torch.Tensor:
        r"""Apply AMSBound variant.

        :param ams_bound: bool. whether to apply AMSBound.
        :param exp_avg_sq: torch.Tensor. exp_avg_sq.
        :param max_exp_avg_sq: Optional[torch.Tensor]. max_exp_avg_sq.
        :param eps: float. epsilon.
        """
        if ams_bound:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            de_nom = max_exp_avg_sq.add(eps)
        else:
            de_nom = exp_avg_sq.add(eps)

        return de_nom.sqrt_().add_(eps)

    @staticmethod
    def debias(beta: float, step: int) -> float:
        r"""Adam-style debias correction. Returns `1.0 - beta ** step`.

        :param beta: float. beta.
        :param step: int. number of step.
        """
        return 1.0 - math.pow(beta, step)  # fmt: skip

    @staticmethod
    def debias_beta(beta: float, step: int) -> float:
        r"""Apply the Adam-style debias correction into beta.

        Simplified version of `\^{beta} = beta * (1.0 - beta ** (step - 1)) / (1.0 - beta ** step)`

        :param beta: float. beta.
        :param step: int. number of step.
        """
        beta_n: float = math.pow(beta, step)
        return (beta_n - beta) / (beta_n - 1.0)  # fmt: skip

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        r"""Apply AdamD variant.

        :param adam_debias: bool. whether to apply AdamD.
        :param step_size: float. step size.
        :param bias_correction1: float. bias_correction.
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
        r"""Get step size for rectify optimizer.

        :param is_rectify: bool. whether to apply rectify-variant.
        :param step: int. number of steps.
        :param lr: float. learning rate.
        :param beta2: float. beta2.
        :param n_sma_threshold: float. SMA threshold.
        :param degenerated_to_sgd: bool. degenerated to SGD.
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

        :param grad: torch.Tensor. gradient.
        :param adanorm: bool. whether to apply AdaNorm.
        :param exp_grad_norm: Optional[torch.Tensor]. exp_grad_norm.
        :param r: float. Optional[float]. momentum (ratio).
        """
        if not adanorm or exp_grad_norm is None:
            return grad

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul_(r).add_(grad_norm, alpha=1.0 - r)

        return grad.mul(exp_grad_norm).div_(grad_norm) if exp_grad_norm > grad_norm else grad

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r"""Get approximation of EMA of squared gradient."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    @staticmethod
    def apply_cautious(update: torch.Tensor, grad: torch.Tensor) -> None:
        r"""Apply the Cautious Optimizer feature.

        :param update: torch.Tensor. update. it'll be masked in in-place manner.
        :param grad: torch.Tensor. gradient.
        """
        mask = (update * grad > 0).to(grad.dtype)
        mask.mul_(mask.numel() / (mask.sum() + 1))
        update.mul_(mask)

    @staticmethod
    def get_stable_adamw_rms(grad: torch.Tensor, exp_avg_sq: torch.Tensor, eps: float = 1e-16) -> float:
        r"""Get StableAdamW RMS.

        :param grad: torch.Tensor. gradient.
        :param exp_avg_sq: torch.Tensor. exp_avg_sq.
        :param eps: float. epsilon.
        """
        return grad.pow(2).div_(exp_avg_sq.clip(min=eps)).mean().sqrt_().clip_(min=1.0).item()

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)') -> None:
        if range_type == '[)' and not low <= x < high:
            raise ValueError(f'[-] {name} must be in the range [{low}, {high})')
        if range_type == '[]' and not low <= x <= high:
            raise ValueError(f'[-] {name} must be in the range [{low}, {high}]')
        if range_type == '(]' and not low < x <= high:
            raise ValueError(f'[-] {name} must be in the range ({low}, {high}]')
        if range_type == '()' and not low < x < high:
            raise ValueError(f'[-] {name} must be in the range ({low}, {high})')

    @staticmethod
    def validate_non_negative(x: Optional[float], name: str) -> None:
        if x is not None and x < 0.0:
            raise ValueError(f'[-] {name} must be non-negative')

    @staticmethod
    def validate_non_positive(x: Optional[float], name: str) -> None:
        if x is not None and x > 0.0:
            raise ValueError(f'[-] {name} must be non-positive')

    @staticmethod
    def validate_positive(x: Union[float, int], name: str) -> None:
        if x <= 0:
            raise ValueError(f'[-] {name} must be positive')

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper') -> None:
        if bound_type == 'upper' and constant > boundary:
            raise ValueError(f'[-] constant {constant} must be in a range of (-inf, {boundary}]')
        if bound_type == 'lower' and constant < boundary:
            raise ValueError(f'[-] constant {constant} must be in a range of [{boundary}, inf)')

    @staticmethod
    def validate_step(step: int, step_type: str) -> None:
        if step < 1:
            raise NegativeStepError(step, step_type=step_type)

    @staticmethod
    def validate_options(x: str, name: str, options: List[str]) -> None:
        if x not in options:
            opts: str = ' or '.join([f'\'{option}\'' for option in options]).strip()
            raise ValueError(f'[-] {name} {x} must be one of ({opts})')

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]) -> None:
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_mod(x: int, y: int) -> None:
        if x % y != 0:
            raise ValueError(f'[-] {x} must be divisible by {y}')

    def validate_betas(self, betas: BETAS) -> None:
        if betas[0] is not None:
            self.validate_range(betas[0], 'beta1', 0.0, 1.0, range_type='[]')

        self.validate_range(betas[1], 'beta2', 0.0, 1.0, range_type='[]')

        if len(betas) < 3:
            return

        if betas[2] is not None:
            self.validate_range(betas[2], 'beta3', 0.0, 1.0, range_type='[]')

    def validate_nus(self, nus: Union[float, Tuple[float, float]]) -> None:
        if isinstance(nus, float):
            self.validate_range(nus, 'nu', 0.0, 1.0, range_type='[]')
        else:
            self.validate_range(nus[0], 'nu1', 0.0, 1.0, range_type='[]')
            self.validate_range(nus[1], 'nu2', 0.0, 1.0, range_type='[]')

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def step(self, closure: CLOSURE = None) -> LOSS:  # pragma: no cover
        raise NotImplementedError
