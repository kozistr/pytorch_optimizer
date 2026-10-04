import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.type import (
    Betas,
    Closure,
    Defaults,
    HutchinsonG,
    Loss,
    OptimizerInstanceOrClass,
    ParamGroup,
    ParamsT,
    State,
)
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt_


class BaseOptimizer(ABC, Optimizer):
    """Shared update, validation, and state helpers for optimizers."""

    state: State

    def __init__(self, params: ParamsT, defaults: Defaults) -> None:
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state while preserving non-floating tensor types and container metadata."""
        super().load_state_dict(state_dict)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for p, key in zip(group['params'], saved_group['params']):
                if key in state_dict['state']:
                    self.state[p] = self._restore_state_types(self.state[p], state_dict['state'][key])

    def _restore_state_types(self, value, saved_value):
        if isinstance(saved_value, torch.Tensor):
            if not saved_value.is_floating_point() and not saved_value.is_complex():
                return saved_value.to(device=value.device)
            return value
        if isinstance(saved_value, str):
            return saved_value
        if isinstance(saved_value, dict):
            return {key: self._restore_state_types(value[key], saved) for key, saved in saved_value.items()}
        if isinstance(saved_value, (tuple, list, deque)):
            restored = [self._restore_state_types(item, saved) for item, saved in zip(value, saved_value)]
            return deque(restored, maxlen=saved_value.maxlen) if isinstance(saved_value, deque) else type(saved_value)(
                restored
            )
        return value

    @staticmethod
    def load_optimizer(optimizer: OptimizerInstanceOrClass, **kwargs) -> Optimizer:
        """Return an optimizer instance or instantiate an optimizer class.

        Args:
            optimizer: Optimizer instance or class.
            **kwargs (dict): Constructor options. Must include `params` when passing a class.

        Returns:
            Optimizer: Existing or newly constructed optimizer.

        Raises:
            ValueError: An optimizer class is supplied without `params`.

        """
        if isinstance(optimizer, Optimizer):
            return optimizer

        if 'params' in kwargs:
            params = kwargs.pop('params')
            return optimizer(params, **kwargs)

        raise ValueError('need to pass `params` when you pass the `torch.optim.Optimizer` instance.')

    @staticmethod
    @torch.no_grad()
    def set_hessian(param_groups: list[ParamGroup], state: State, hessian: list[torch.Tensor]) -> None:
        """Store externally computed Hessian estimates in the optimizer state.

        Args:
            param_groups: Optimizer parameter groups.
            state: Optimizer state dictionary to update.
            hessian: One tensor per parameter, in parameter group order, with matching shapes.

        Raises:
            ValueError: A Hessian tensor and its parameter have different shapes.

        Examples:
            ```python
            BaseOptimizer.set_hessian(optimizer.param_groups, optimizer.state, hessian)
            ```

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
    def zero_hessian(param_groups: list[ParamGroup], state: State, pre_zero: bool = True) -> None:
        """Initialize Hessian buffers and optionally clear existing estimates.

        Args:
            param_groups: Parameter groups from the optimizer.
            state: Optimizer state dictionary.
            pre_zero: Clear existing Hessian estimates before accumulating new ones.

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
        param_groups: list[ParamGroup],
        state: State,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: HutchinsonG = 'gaussian',
    ) -> None:
        """Accumulate Hutchinson estimates of the Hessian diagonal in the optimizer state.

        Args:
            param_groups: Parameter groups from the optimizer.
            state: Optimizer state dictionary.
            num_samples: Number of noise vectors for the Hessian diagonal estimate.
            alpha: Scale of the estimate to add to existing Hessian buffers.
            distribution: Noise distribution: `'gaussian'` or `'rademacher'`.

        """
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params: list[torch.Tensor] = [
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
                zs = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in params]
            else:
                zs = [torch.randn_like(p) for p in params]

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, retain_graph=i < num_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                state[p]['hessian'].add_(h_z * z, alpha=alpha / num_samples)

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: torch.Tensor | None,
        lr: float | torch.Tensor,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: float | None = None,
    ) -> None:
        """Apply coupled or decoupled weight decay in place.

        Args:
            p: Parameter tensor to apply weight decay to.
            grad: Gradient to modify for coupled decay. May be `None` for decoupled decay.
            lr: Learning rate to scale the update.
            weight_decay: Weight decay coefficient (L2 penalty).
            weight_decouple: If True, applies decoupled weight decay as in AdamW.
            fixed_decay: If True, fixes weight decay to not depend on learning rate.
            ratio: Optional scaling factor for decoupled weight decay.

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
        """Decay parameter entries whose signs agree with the update, in place.

        Args:
            p: Parameter tensor to apply weight decay to.
            update: Update tensor.
            lr: Learning rate to scale the update.
            weight_decay: Weight decay coefficient (L2 penalty).

        """
        p.copy_(torch.where(update * p >= 0, p * (1.0 - weight_decay * lr), p))

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: torch.Tensor | None,
        eps: float,
        exp_avg_sq_eps: float = 1e-15,
    ) -> torch.Tensor:
        """Compute an adaptive denominator, with optional running maximum second moments.

        Args:
            ams_bound: Whether to apply the AMSBound variant.
            exp_avg_sq: Exponential moving average of squared gradients.
            max_exp_avg_sq: Running elementwise maximum of `exp_avg_sq`, updated in place for AMSBound.
            eps: Small epsilon value for numerical stability.
            exp_avg_sq_eps: Epsilon used specifically for numerical stability in exp_avg_sq computations.

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
        """Return the moment bias correction factor `1 - beta ** step`.

        Args:
            beta: Exponential decay rate for moment estimates.
            step: Current optimization step number.

        """
        return 1.0 - math.pow(beta, step)  # fmt: skip

    @staticmethod
    def debias_beta(beta: float, step: int) -> float:
        """Return the decay rate for a bias corrected moving average.

        Computes `beta * (1 - beta ** (step - 1)) / (1 - beta ** step)`.

        Args:
            beta: Exponential decay rate.
            step: Optimization step, starting at 1.

        Returns:
            float: Bias corrected decay rate.

        """
        beta_n: float = beta ** step
        return (beta_n - beta) / (beta_n - 1.0)  # fmt: skip

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        """Apply AdamD variant.

        Args:
            adam_debias: If True, only corrects the denominator to avoid inflating step sizes early in training.
            step_size: The step size for the update.
            bias_correction1: The bias correction factor for the first moment.

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
    ) -> tuple[float, float]:
        """Compute the RAdam step size and effective moving average length.

        Args:
            is_rectify: Whether to apply the rectify variant.
            step: Current step number.
            lr: Base learning rate.
            beta2: Decay rate of the second moment.
            n_sma_threshold: Simple Moving Average (SMA) threshold for rectification.
            degenerated_to_sgd: Whether to degenerate to SGD if below threshold.

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
        grad: torch.Tensor, adanorm: bool, exp_grad_norm: torch.Tensor | None = None, r: float | None = 0.95
    ) -> torch.Tensor:
        """Rescale gradients whose norm falls below its running average.

        Args:
            grad: Gradient.
            adanorm: Whether to use the AdaNorm variant.
            exp_grad_norm: Exponential moving average of gradient norm.
            r: Decay rate for the gradient norm average.

        """
        if not adanorm or exp_grad_norm is None:
            return grad

        if r is None:
            r = 0.95

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul_(r).add_(grad_norm, alpha=1.0 - r)

        if grad_norm > 0 and exp_grad_norm > grad_norm:
            return grad.mul(exp_grad_norm).div_(grad_norm)

        return grad

    @staticmethod
    def get_rms(x: Sequence[torch.Tensor] | torch.Tensor) -> Sequence[torch.Tensor] | torch.Tensor:
        """Compute the root mean square of a tensor or each tensor in a list."""
        if isinstance(x, torch.Tensor):
            return x.norm(2).div_(math.sqrt(x.numel()))

        factors: list[float] = [math.sqrt(p.numel()) for p in x]
        norms = torch._foreach_norm(x, ord=2)
        torch._foreach_div_(norms, factors)

        return norms

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: list[torch.Tensor] | torch.Tensor,
        exp_avg_sq_col: list[torch.Tensor] | torch.Tensor,
        output: list[torch.Tensor] | torch.Tensor,
    ) -> None:
        """Write a factored inverse root second moment approximation to `output`.

        Args:
            exp_avg_sq_row: Row second moments, as a tensor or list of tensors.
            exp_avg_sq_col: Corresponding column second moments.
            output: Tensor or list of tensors to update in place.

        """
        if isinstance(exp_avg_sq_row, torch.Tensor):
            r_factor: torch.Tensor = (
                (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
            )
            c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
            torch.mul(r_factor, c_factor, out=output)
            return

        row_means = [r.mean(dim=-1, keepdim=True) for r in exp_avg_sq_row]

        r_factors = torch._foreach_div(exp_avg_sq_row, row_means)
        foreach_rsqrt_(r_factors)
        r_factors = [r_factor.unsqueeze(-1) for r_factor in r_factors]

        c_factors = [c_factor.unsqueeze(-2).rsqrt() for c_factor in exp_avg_sq_col]

        torch._foreach_copy_(output, torch._foreach_mul(r_factors, c_factors))

    @staticmethod
    def apply_cautious(update: torch.Tensor, grad: torch.Tensor) -> None:
        """Mask updates that disagree with the gradient sign and rescale them in place.

        Args:
            update: Update tensor, masked in place.
            grad: Gradient tensor.

        """
        mask = (update * grad > 0).to(grad.dtype)
        mask.mul_(mask.numel() / (mask.sum() + 1))
        update.mul_(mask)

    @staticmethod
    @torch.no_grad()
    def apply_orthogonal_gradients(params: Iterable[torch.Tensor], eps: float = 1e-16) -> None:
        """Project gradients orthogonally to parameters and restore their norms.

        Args:
            params: Parameters whose dense real gradients are modified in place.
            eps: Small value to prevent division by zero.

        """
        for p in params:
            if p.grad is None or p.grad.is_sparse or torch.is_complex(p):
                continue

            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            w = p.reshape(-1).to(dtype=dtype)
            g = p.grad.reshape(-1).to(dtype=dtype)

            proj = torch.dot(w, g).div_(torch.dot(w, w).add_(eps))
            g_ortho = g.sub(w * proj)

            g_norm = g.norm(2)
            g_ortho_norm = g_ortho.norm(2)
            rounding_eps = 4.0 * torch.finfo(dtype).eps
            # A parallel gradient has no orthogonal direction to normalize.
            g_ortho.masked_fill_(g_ortho_norm <= rounding_eps * g_norm, 0.0)
            g_ortho.mul_(g_norm / g_ortho_norm.add_(eps))

            p.grad.copy_(g_ortho.view_as(p.grad))

    @staticmethod
    def can_use_foreach(group: ParamGroup, foreach: bool | None) -> bool:
        """Check whether a parameter group supports batched tensor updates.

        Args:
            group: Parameter group to inspect.
            foreach: Set to `False` to disable batched updates. `True` and `None` check tensor compatibility.

        Returns:
            bool: `True` if at least one parameter has a gradient and all such parameters
                are real with dense gradients.

        """
        if foreach is False:
            return False

        has_param: bool = False
        for p in group['params']:
            g = p.grad
            if g is None:
                continue

            has_param = True
            if g.is_sparse or torch.is_complex(p):
                return False

        return has_param

    @staticmethod
    def collect_trainable_params(
        group: ParamGroup,
        state: State,
        state_keys: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Collect parameters with gradients and their requested state tensors.

        Args:
            group: Optimizer parameter group.
            state: Optimizer state dictionary.
            state_keys: State entries to collect. `None` collects no state tensors.

        Returns:
            tuple: Parameters, gradients, and a dictionary of lists of available state tensors.

        """
        if state_keys is None:
            state_keys = []

        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        state_dict: dict[str, list[torch.Tensor]] = {key: [] for key in state_keys}

        for p in group['params']:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)

            if state_keys:
                p_state = state[p]
                for key in state_keys:
                    if key in p_state:
                        state_dict[key].append(p_state[key])

        return params, grads, state_dict

    @staticmethod
    def apply_weight_decay_foreach(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        lr: list[float] | list[torch.Tensor] | tuple[torch.Tensor, ...] | float | torch.Tensor,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
    ) -> None:
        """Apply weight decay to a list of parameters.

        Args:
            params: List of parameter tensors.
            grads: List of gradient tensors.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            weight_decouple: If True, applies decoupled weight decay as in AdamW.
            fixed_decay: If True, fixes weight decay to not depend on learning rate.

        """
        if weight_decay == 0.0:
            return

        if not weight_decouple:
            torch._foreach_add_(grads, params, alpha=weight_decay)
            return

        if fixed_decay:
            factor = 1.0 - weight_decay
        elif isinstance(lr, Sequence):
            factor = torch._foreach_mul(lr, -weight_decay)
            torch._foreach_add_(factor, 1.0)
        else:
            factor = 1.0 - weight_decay * lr

        torch._foreach_mul_(params, factor)

    @staticmethod
    def get_stable_adamw_rms(grad: torch.Tensor, exp_avg_sq: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        """Get StableAdamW RMS as a scalar tensor on the gradient's device.

        Args:
            grad: Gradient.
            exp_avg_sq: Exponential moving average of squared gradient.
            eps: Small value to prevent division by zero.

        Returns:
            torch.Tensor: RMS scale, computed in float32 for float16 and bfloat16 gradients.

        """
        if grad.dtype in (torch.float16, torch.bfloat16):
            grad, exp_avg_sq = grad.float(), exp_avg_sq.float()

        return grad.pow(2).div_(exp_avg_sq.clip(min=eps)).mean().sqrt_().clip_(min=1.0)

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)') -> None:
        """Raise `ValueError` if a value falls outside the requested interval."""
        if range_type == '[)' and not low <= x < high:
            raise ValueError(f'{name} must be in the range [{low}, {high})')
        if range_type == '[]' and not low <= x <= high:
            raise ValueError(f'{name} must be in the range [{low}, {high}]')
        if range_type == '(]' and not low < x <= high:
            raise ValueError(f'{name} must be in the range ({low}, {high}]')
        if range_type == '()' and not low < x < high:
            raise ValueError(f'{name} must be in the range ({low}, {high})')

    @staticmethod
    def validate_non_negative(x: float | None, name: str) -> None:
        """Raise `ValueError` for a negative value. Accept `None`."""
        if x is not None and x < 0.0:
            raise ValueError(f'{name} must be non-negative')

    @staticmethod
    def validate_non_positive(x: float | None, name: str) -> None:
        """Raise `ValueError` for a positive value. Accept `None`."""
        if x is not None and x > 0.0:
            raise ValueError(f'{name} must be non-positive')

    @staticmethod
    def validate_positive(x: float | int, name: str) -> None:
        """Raise `ValueError` if a value is zero or negative."""
        if x <= 0:
            raise ValueError(f'{name} must be positive')

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper') -> None:
        """Raise `ValueError` if a value exceeds an upper bound or falls below a lower bound."""
        if bound_type == 'upper' and constant > boundary:
            raise ValueError(f'constant {constant} must be in a range of (-inf, {boundary}]')
        if bound_type == 'lower' and constant < boundary:
            raise ValueError(f'constant {constant} must be in a range of [{boundary}, inf)')

    @staticmethod
    def validate_step(step: int, step_type: str) -> None:
        """Raise `NegativeStepError` if the step number is less than 1."""
        if step < 1:
            raise NegativeStepError(step, step_type=step_type)

    @staticmethod
    def validate_options(x: str, name: str, options: list[str]) -> None:
        """Raise `ValueError` if an option is outside the allowed choices."""
        if x not in options:
            opts: str = ' or '.join([f"'{option}'" for option in options]).strip()
            raise ValueError(f'{name} {x} must be one of ({opts})')

    @staticmethod
    def validate_learning_rate(learning_rate: float | torch.Tensor | None) -> None:
        """Raise `NegativeLRError` for a negative learning rate. Accept `None`."""
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_mod(x: int, y: int) -> None:
        """Raise `ValueError` if `x` is not divisible by `y`."""
        if x % y != 0:
            raise ValueError(f'{x} must be divisible by {y}')

    def validate_betas(
        self,
        betas: Betas | tuple[None, float],
        beta_range_type: str = '[)',
        beta3_range_type: str = '[]',
    ) -> None:
        """Validate the first two beta values and an optional third value against their intervals."""
        if betas[0] is not None:
            self.validate_range(betas[0], 'beta1', 0.0, 1.0, range_type=beta_range_type)

        self.validate_range(betas[1], 'beta2', 0.0, 1.0, range_type=beta_range_type)

        if len(betas) < 3:
            return

        if betas[2] is not None:
            self.validate_range(betas[2], 'beta3', 0.0, 1.0, range_type=beta3_range_type)

    def validate_nus(self, nus: float | tuple[float, float]) -> None:
        """Validate one or two quasi-hyperbolic discount factors in `[0, 1]`."""
        if isinstance(nus, tuple):
            nu1, nu2 = nus
            self.validate_range(nu1, 'nu1', 0.0, 1.0, range_type='[]')
            self.validate_range(nu2, 'nu2', 0.0, 1.0, range_type='[]')
        else:
            self.validate_range(nus, 'nu', 0.0, 1.0, range_type='[]')

    @abstractmethod
    def init_group(self, group: ParamGroup, **kwargs) -> None:  # pragma: no cover
        """Initialize optimizer state for a parameter group."""
        return

    @staticmethod
    def view_as_real(param, *state_and_grads) -> tuple:
        """Return real views of complex parameters, gradients, and state tensors."""
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            state_and_grads = tuple(
                torch.view_as_real(s) if (s is not None and torch.is_complex(s)) else s if s is not None else None
                for s in state_and_grads
            )

        return param, *state_and_grads

    @staticmethod
    def maximize_gradient(grad: torch.Tensor, maximize: bool = False) -> None:
        """Negate the gradient in place when maximizing the objective."""
        if maximize:
            grad.neg_()

    def step(self, closure: Closure = None) -> Loss:  # pragma: no cover
        raise NotImplementedError
