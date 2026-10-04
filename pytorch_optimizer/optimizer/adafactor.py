import math
from collections.abc import Sequence

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt


class AdaFactor(BaseOptimizer):
    """Factored adaptive updates with the BigVision AdaFactor variant.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Update momentum decay and second moment decay cap. Set the first value to `None` to disable
            momentum.
        decay_rate: Exponent controlling the step-dependent second moment decay.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        clip_threshold: Maximum root mean square of the preconditioned update.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        scale_parameter: If True, the learning rate is scaled by root mean square of parameter.
        relative_step: If True, time dependent learning rate is computed instead of external learning rate.
        warmup_init: Warm up the relative step size from `1e-6 * step`.
        eps1: Stability constant added to squared gradients.
        eps2: Lower bound for parameter RMS scaling.
        momentum_dtype: Data type for the optional momentum buffer.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | None = 1e-3,
        betas: tuple[None, float] | tuple[float, float] | tuple[float, float, float] = (0.9, 0.999),
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        momentum_dtype: torch.dtype = torch.bfloat16,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold
        self.eps1: float = eps1 if momentum_dtype != torch.float16 else 1e-7
        self.eps2 = eps2
        self.momentum_dtype = momentum_dtype
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'scale_parameter': scale_parameter,
            'relative_step': relative_step,
            'warmup_init': warmup_init,
            'eps1': eps1,
            'eps2': eps2,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaFactor'

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for p, key in zip(group['params'], saved_group['params']):
                saved_state = state_dict['state'].get(key, {})
                if 'exp_avg' in saved_state:
                    self.state[p]['exp_avg'] = saved_state['exp_avg'].to(device=p.device, dtype=self.momentum_dtype)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        beta1: float = kwargs.get('beta1', 0.9)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            grad_shape: tuple[int, ...] = grad.shape
            factored: bool = self.get_options(grad_shape)

            if len(state) == 0:
                state['RMS'] = 0.0

                if beta1 is not None:
                    state['exp_avg'] = torch.zeros_like(p, dtype=self.momentum_dtype)

                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                if group['ams_bound']:
                    state['exp_avg_sq_hat'] = torch.zeros_like(grad)

    @staticmethod
    def get_relative_step_size(lr: float, step: int, relative_step: bool, warmup_init: bool) -> float:
        if not relative_step:
            return lr

        min_step: float = 1e-6 * step if warmup_init else 1e-2
        return min(min_step, 1.0 / math.sqrt(step))

    def get_lr(
        self,
        relative_step_size: torch.Tensor | float,
        rms: Sequence[torch.Tensor] | torch.Tensor | float,
        scale_parameter: bool,
    ) -> Sequence[torch.Tensor] | torch.Tensor | float:
        """Compute effective learning rates with optional parameter RMS scaling."""
        if not scale_parameter:
            return relative_step_size

        if not isinstance(rms, Sequence):
            return max(self.eps2, rms) * relative_step_size

        lrs = torch._foreach_maximum(rms, self.eps2)
        torch._foreach_mul_(lrs, relative_step_size)

        return lrs

    @staticmethod
    def get_options(shape: tuple[int, ...]) -> bool:
        """Return whether the gradient supports factored second moments."""
        return len(shape) >= 2

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        if group.get('cautious'):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sq_rows: list[torch.Tensor],
        exp_avg_sq_cols: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        exp_avg_sq_hats: list[torch.Tensor],
        beta1: float,
        beta2_t: float,
        relative_step_size: float,
    ) -> None:
        bias_correction2: float = 1.0 - beta2_t

        if self.maximize:
            torch._foreach_neg_(grads)

        rms_values = self.get_rms(params)
        lrs = self.get_lr(relative_step_size, rms_values, group['scale_parameter'])

        updates = torch._foreach_pow(grads, 2)
        torch._foreach_add_(updates, self.eps1)

        factored_offsets, non_factored_offsets = [], []
        factored_updates, non_factored_updates = [], []
        for i, grad in enumerate(grads):
            if self.get_options(grad.shape):
                factored_updates.append(updates[i])
                factored_offsets.append(i)
            else:
                non_factored_updates.append(updates[i])
                non_factored_offsets.append(i)

        if factored_updates:
            row_means, col_means = [], []
            for factored_update in factored_updates:
                row_means.append(factored_update.mean(dim=-1))
                col_means.append(factored_update.mean(dim=-2))

            torch._foreach_lerp_(exp_avg_sq_rows, row_means, weight=bias_correction2)
            torch._foreach_lerp_(exp_avg_sq_cols, col_means, weight=bias_correction2)

            self.approximate_sq_grad(exp_avg_sq_rows, exp_avg_sq_cols, factored_updates)

        if non_factored_updates:
            torch._foreach_lerp_(exp_avg_sqs, non_factored_updates, weight=bias_correction2)

            non_factored_updates = foreach_rsqrt(exp_avg_sqs)

        updates = [None] * len(grads)

        for offset, update in zip(factored_offsets, factored_updates):
            updates[offset] = update

        for offset, update in zip(non_factored_offsets, non_factored_updates):
            updates[offset] = update

        if group['ams_bound']:
            inv_updates = torch._foreach_reciprocal(updates)
            torch._foreach_maximum_(exp_avg_sq_hats, inv_updates)

            updates = foreach_rsqrt(torch._foreach_div(exp_avg_sq_hats, bias_correction2))

        torch._foreach_mul_(updates, grads)

        rms_values = self.get_rms(updates)
        torch._foreach_div_(rms_values, self.clip_threshold)
        torch._foreach_clamp_min_(rms_values, 1.0)

        torch._foreach_div_(updates, rms_values)
        torch._foreach_mul_(updates, lrs)

        if beta1 is not None:
            is_dtype_different: bool = self.momentum_dtype != grads[0].dtype
            if is_dtype_different:
                updates = [update.to(self.momentum_dtype) for update in updates]

            torch._foreach_lerp_(exp_avgs, updates, weight=1.0 - beta1)

            if is_dtype_different:
                updates = [exp_avg.to(grads[0].dtype) for exp_avg in exp_avgs]
            else:
                torch._foreach_copy_(updates, exp_avgs)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lrs,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        torch._foreach_sub_(params, updates)

    def _step_per_param(self, group: ParamGroup, beta1: float, beta2_t: float, relative_step_size: float) -> None:
        bias_correction2: float = 1.0 - beta2_t
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            factored: bool = self.get_options(grad.shape)

            state['RMS'] = self.get_rms(p)

            lr = self.get_lr(relative_step_size, state['RMS'], group['scale_parameter'])

            # NOTE(kozistr): adding `eps1` here instead of clipping max by eps1 later
            update = grad.square().add_(self.eps1)

            if factored:
                exp_avg_sq_row, exp_avg_sq_col = state['exp_avg_sq_row'], state['exp_avg_sq_col']

                exp_avg_sq_row.lerp_(update.mean(dim=-1), weight=bias_correction2)
                exp_avg_sq_col.lerp_(update.mean(dim=-2), weight=bias_correction2)

                self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
            else:
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.lerp_(update, weight=bias_correction2)
                torch.rsqrt(exp_avg_sq, out=update)

            if group['ams_bound']:
                exp_avg_sq_hat = state['exp_avg_sq_hat']
                torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
                torch.rsqrt(exp_avg_sq_hat / bias_correction2, out=update)

            update.mul_(grad)

            factor = self.get_rms(update).div_(self.clip_threshold).clamp_min_(1.0)
            update.div_(factor).mul_(lr)

            if beta1 is not None:
                exp_avg = state['exp_avg']
                if self.momentum_dtype != grad.dtype:
                    exp_avg.lerp_(update.to(self.momentum_dtype), weight=1.0 - beta1)
                    update = exp_avg.to(grad.dtype)
                else:
                    exp_avg.lerp_(update, weight=1.0 - beta1)
                    update = exp_avg.clone()

                if group.get('cautious'):
                    self.apply_cautious(update, grad)

            self.apply_weight_decay(
                p=p,
                grad=None,
                lr=lr,
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            p.add_(-update)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2_cap = group['betas']

            self.init_group(group, beta1=beta1)
            group['step'] += 1

            beta2_t: float = min(beta2_cap, 1.0 - math.pow(group['step'], self.decay_rate))

            relative_step_size: float = self.get_relative_step_size(
                lr=group['lr'],
                step=group['step'],
                relative_step=group['relative_step'],
                warmup_init=group['warmup_init'],
            )

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group,
                    self.state,
                    state_keys=['exp_avg', 'exp_avg_sq_row', 'exp_avg_sq_col', 'exp_avg_sq', 'exp_avg_sq_hat'],
                )
                if params:
                    self._step_foreach(
                        group,
                        params,
                        grads,
                        state_dict['exp_avg'],
                        state_dict['exp_avg_sq_row'],
                        state_dict['exp_avg_sq_col'],
                        state_dict['exp_avg_sq'],
                        state_dict['exp_avg_sq_hat'],
                        beta1,
                        beta2_t,
                        relative_step_size,
                    )
            else:
                self._step_per_param(group, beta1, beta2_t, relative_step_size)

        return loss
