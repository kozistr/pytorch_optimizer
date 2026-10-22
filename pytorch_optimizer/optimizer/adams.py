import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_addcdiv_,
    group_tensors_by_device_and_dtype,
)


class AdamS(BaseOptimizer):
    """Adam with stable weight decay.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-4,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach
        self._compiled_update_moments: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamS'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_update_moments = compile_foreach_step(self._update_moments_foreach, compile_kwargs)
        super()._compile_foreach(compile_kwargs)

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

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    def _update_moments_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> torch.Tensor:
        beta1, beta2 = group['betas']
        if self.maximize:
            torch._foreach_neg_(grads)
        if not group['weight_decouple']:
            self.apply_weight_decay_foreach(
                params, grads, group['lr'], group['weight_decay'], False, group['fixed_decay']
            )
        torch._foreach_lerp_(state['exp_avg'], grads, weight=1.0 - beta1)
        torch._foreach_mul_(state['exp_avg_sq'], beta2)
        torch._foreach_addcmul_(state['exp_avg_sq'], grads, grads, value=1.0 - beta2)
        moments = state['exp_avg_sq']
        if group['ams_bound']:
            torch._foreach_maximum_(state['max_exp_avg_sq'], moments)
            moments = state['max_exp_avg_sq']
        return torch.stack([moment.sum() / bias_correction2 for moment in moments]).sum()

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
        mean_square: torch.Tensor,
        step_size: float | torch.Tensor,
    ) -> None:
        exp_avg_sq_hat_mean = mean_square.to(dtype=torch.promote_types(mean_square.dtype, torch.float32)).sqrt()
        exp_avg_sq_hat_mean.add_(self.defaults['eps'])
        if group['weight_decouple'] and group['weight_decay'] > 0.0:
            decay = group['weight_decay'] * (1.0 if group['fixed_decay'] else group['lr'])
            torch._foreach_mul_(params, 1.0 - decay / exp_avg_sq_hat_mean)
        moments = state['max_exp_avg_sq'] if group['ams_bound'] else state['exp_avg_sq']
        de_noms = torch._foreach_div(moments, bias_correction2)
        torch._foreach_sqrt_(de_noms)
        torch._foreach_add_(de_noms, group['eps'])
        foreach_addcdiv_(params, state['exp_avg'], de_noms, value=-step_size)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        exp_avg_sq_hat_sum: float | torch.Tensor = 0.0
        foreach_batches: list[list[dict]] = []

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            batches = []
            if not group.get('adanorm') and self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state,
                    state_keys=['exp_avg', 'exp_avg_sq'] + (['max_exp_avg_sq'] if group['ams_bound'] else []),
                )
                batches = group_tensors_by_device_and_dtype(params, grads, state)
            foreach_batches.append(batches)
            if batches:
                param_size += sum(p.numel() for p in params)
                update_moments = self._compiled_update_moments or self._update_moments_foreach
                for batch in batches:
                    exp_avg_sq_hat_sum += update_moments(
                        group, batch['params'], batch['grads'], batch, bias_correction2
                    )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                param_size += p.numel()

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if not group['weight_decouple']:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=False,
                        fixed_decay=group['fixed_decay'],
                    )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.lerp_(s_grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group['ams_bound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    exp_avg_sq_hat = max_exp_avg_sq
                else:
                    exp_avg_sq_hat = exp_avg_sq

                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError

        mean_square = exp_avg_sq_hat_sum / param_size

        for group, batches in zip(self.param_groups, foreach_batches):
            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'],
                bias_correction1=bias_correction1,
            )

            if batches:
                for batch in batches:
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch,
                        bias_correction2, mean_square, step_size,
                    )
                continue

            exp_avg_sq_hat_mean = math.sqrt(mean_square) + self.defaults['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=1.0 / exp_avg_sq_hat_mean,
                )

                exp_avg_sq_hat = state['max_exp_avg_sq'] if group['ams_bound'] else state['exp_avg_sq']
                de_nom = exp_avg_sq_hat.div(bias_correction2).sqrt_().add_(group['eps'])

                p.addcdiv_(state['exp_avg'], de_nom, value=-step_size)

        return loss
