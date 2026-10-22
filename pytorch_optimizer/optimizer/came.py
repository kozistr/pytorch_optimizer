import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    foreach_rsqrt_,
    group_tensors_by_device_and_dtype,
)


class CAME(BaseOptimizer):
    """Factored adaptive updates with confidence weighted momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for gradient momentum, squared gradients, and squared update residuals.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        clip_threshold: Maximum root mean square of the preconditioned update.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps1: Stability constant added to squared gradients.
        eps2: Stability constant added to squared update residuals.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
            Float16 updates remain eager when compilation is requested.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 2e-4,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2
        self.maximize = maximize
        self.foreach = foreach
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps1': eps1,
            'eps2': eps2,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CAME'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

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
                state['exp_avg'] = torch.zeros_like(p)

                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                    state['exp_avg_res_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_res_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                if group['ams_bound']:
                    state['exp_avg_sq_hat'] = torch.zeros_like(grad)

                state['RMS'] = 0.0

    @staticmethod
    def get_options(shape: tuple[int, ...]) -> bool:
        """Return whether the gradient supports factored second moments."""
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> torch.Tensor:
        """Compute the root mean square of a tensor."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ):
        """Write a factored inverse root second moment approximation to `output`."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()

        torch.mul(r_factor, c_factor, out=output)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        beta1, beta2, beta3 = group['betas']
        factored = 'exp_avg_sq_row' in state

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        updates = torch._foreach_mul(grads, grads)
        torch._foreach_add_(updates, self.eps1)

        if factored:
            rows, cols = state['exp_avg_sq_row'], state['exp_avg_sq_col']

            torch._foreach_mul_(rows, beta2)
            torch._foreach_add_(rows, [update.mean(dim=-1) for update in updates], alpha=1.0 - beta2)

            torch._foreach_mul_(cols, beta2)
            torch._foreach_add_(cols, [update.mean(dim=-2) for update in updates], alpha=1.0 - beta2)

            for row, col, update in zip(rows, cols, updates):
                self.approximate_sq_grad(row, col, update)
        else:
            torch._foreach_mul_(state['exp_avg_sq'], beta2)
            torch._foreach_add_(state['exp_avg_sq'], updates, alpha=1.0 - beta2)

            updates = [moment.rsqrt() for moment in state['exp_avg_sq']]

        if group['ams_bound']:
            torch._foreach_maximum_(state['exp_avg_sq_hat'], [1.0 / update for update in updates])
            updates = torch._foreach_div(state['exp_avg_sq_hat'], beta2)
            foreach_rsqrt_(updates)

        torch._foreach_mul_(updates, grads)

        clipping = [(self.get_rms(update) / self.clip_threshold).clamp_(min=1.0) for update in updates]
        torch._foreach_div_(updates, clipping)

        torch._foreach_lerp_(state['exp_avg'], updates, weight=1.0 - beta1)

        if factored:
            residuals = torch._foreach_sub(updates, state['exp_avg'])
            torch._foreach_pow_(residuals, 2)
            torch._foreach_add_(residuals, self.eps2)

            rows, cols = state['exp_avg_res_row'], state['exp_avg_res_col']

            torch._foreach_mul_(rows, beta3)
            torch._foreach_add_(rows, [res.mean(dim=-1) for res in residuals], alpha=1.0 - beta3)

            torch._foreach_mul_(cols, beta3)
            torch._foreach_add_(cols, [res.mean(dim=-2) for res in residuals], alpha=1.0 - beta3)

            for row, col, update in zip(rows, cols, updates):
                self.approximate_sq_grad(row, col, update)

            torch._foreach_mul_(updates, state['exp_avg'])
        else:
            updates = state['exp_avg']

        foreach_add_(params, updates, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2, beta3 = group['betas']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                for factored in (False, True):
                    keys = ['exp_avg'] + (
                        ['exp_avg_sq_row', 'exp_avg_sq_col', 'exp_avg_res_row', 'exp_avg_res_col']
                        if factored else ['exp_avg_sq']
                    ) + (['exp_avg_sq_hat'] if group['ams_bound'] else [])

                    active = {'params': [p for p in group['params'] if self.get_options(p.shape) == factored]}
                    params, grads, state = self.collect_trainable_params(active, self.state, state_keys=keys)

                    for batch in group_tensors_by_device_and_dtype(params, grads, state):
                        rms = torch._foreach_norm(batch['params'])
                        torch._foreach_div_(rms, [math.sqrt(param.numel()) for param in batch['params']])

                        for param, value in zip(batch['params'], rms):
                            self.state[param]['RMS'] = value

                        update = self._step_foreach
                        if self._compiled_step_foreach is not None and batch['params'][0].dtype != torch.float16:
                            update = self._compiled_step_foreach

                        update(group, batch['params'], batch['grads'], batch)

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_shape: tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state['RMS'] = self.get_rms(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row, exp_avg_sq_col = state['exp_avg_sq_row'], state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    torch.rsqrt(exp_avg_sq, out=update)

                if group['ams_bound']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']
                    torch.max(exp_avg_sq_hat, 1 / update, out=exp_avg_sq_hat)
                    torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

                update.mul_(grad)

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                exp_avg = state['exp_avg']
                exp_avg.lerp_(update, weight=1.0 - beta1)

                res = update - exp_avg
                res.pow_(2).add_(self.eps2)

                if factored:
                    exp_avg_res_row, exp_avg_res_col = state['exp_avg_res_row'], state['exp_avg_res_col']

                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

                    self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, update)
                    update.mul_(exp_avg)
                else:
                    update = exp_avg

                p.add_(update, alpha=-group['lr'])

        return loss
