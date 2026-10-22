import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    group_tensors_by_device_and_dtype,
)


class Adai(BaseOptimizer):
    """SGD with gradient dependent momentum and optional stable weight decay.

    Compiled foreach updates keep float16 and bfloat16 arithmetic eager to preserve update rounding.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Adaptive momentum scaling coefficient and squared gradient decay rate.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        stable_weight_decay: Perform stable weight decay.
        dampening: Dampening factor for momentum.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.1, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        stable_weight_decay: bool = False,
        dampening: float = 1.0,
        eps: float = 1e-3,
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
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'stable_weight_decay': stable_weight_decay,
            'dampening': dampening,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adai'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_update_moments = compile_foreach_step(self._update_moments_foreach, compile_kwargs)
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

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['beta1_prod'] = torch.ones_like(p)

    def _update_moments_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        bias_correction2: float | torch.Tensor,
    ) -> torch.Tensor:
        if self.maximize:
            torch._foreach_neg_(grads)
        if group.get('use_gc'):
            for grad in grads:
                centralize_gradient(grad, gc_conv_only=False)
        if not group['stable_weight_decay']:
            self.apply_weight_decay_foreach(
                params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
            )
        torch._foreach_mul_(exp_avg_sqs, group['betas'][1])
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - group['betas'][1])
        return torch.stack([moment.sum() / bias_correction2 for moment in exp_avg_sqs]).sum()

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
        exp_avg_sq_hat_mean: torch.Tensor,
    ) -> None:
        beta0 = group['betas'][0]
        if group['stable_weight_decay']:
            self.apply_weight_decay_foreach(
                params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
            )

        beta1s = torch._foreach_div(state['exp_avg_sq'], bias_correction2)
        torch._foreach_div_(beta1s, exp_avg_sq_hat_mean)
        torch._foreach_pow_(beta1s, 1.0 / (3.0 - 2.0 * group['dampening']))
        torch._foreach_mul_(beta1s, beta0)
        beta1s = [(1.0 - beta1).clamp_(0.0, 1.0 - group['eps']) for beta1 in beta1s]
        beta3s = [(1.0 - beta1).pow_(group['dampening']) for beta1 in beta1s]
        torch._foreach_mul_(state['beta1_prod'], beta1s)
        torch._foreach_mul_(state['exp_avg'], beta1s)
        torch._foreach_addcmul_(state['exp_avg'], beta3s, grads)
        updates = torch._foreach_div(state['exp_avg'], [1.0 - product for product in state['beta1_prod']])
        torch._foreach_mul_(updates, math.pow(beta0, 1.0 - group['dampening']))
        foreach_add_(params, updates, alpha=-group['lr'])

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

            _, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            batches = []
            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq', 'beta1_prod']
                )
                batches = group_tensors_by_device_and_dtype(params, grads, state)
            foreach_batches.append(batches)
            if batches:
                param_size += sum(p.numel() for p in params)
                for batch in batches:
                    update_moments = (
                        self._compiled_update_moments
                        if self._compiled_update_moments is not None
                        and batch['params'][0].dtype not in (torch.float16, torch.bfloat16)
                        else self._update_moments_foreach
                    )
                    exp_avg_sq_hat_sum += update_moments(
                        group, batch['params'], batch['grads'], batch['exp_avg_sq'], bias_correction2
                    )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                param_size += p.numel()

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if group.get('use_gc'):
                    centralize_gradient(grad, gc_conv_only=False)

                if not group['stable_weight_decay'] and group['weight_decay'] > 0.0:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError

        exp_avg_sq_hat_mean = (exp_avg_sq_hat_sum / param_size).clamp_min_(self.defaults['eps'])

        for group, batches in zip(self.param_groups, foreach_batches):
            beta0, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            if batches:
                for batch in batches:
                    update = (
                        self._compiled_step_foreach
                        if self._compiled_step_foreach is not None
                        and batch['params'][0].dtype not in (torch.float16, torch.bfloat16)
                        else self._step_foreach
                    )
                    update(
                        group, batch['params'], batch['grads'], batch, bias_correction2, exp_avg_sq_hat_mean
                    )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if group['stable_weight_decay'] and group['weight_decay'] > 0.0:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (
                    1.0
                    - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow_(1.0 / (3.0 - 2.0 * group['dampening'])).mul_(beta0)
                ).clamp_(0.0, 1.0 - group['eps'])
                beta3 = (1.0 - beta1).pow_(group['dampening'])

                beta1_prod = state['beta1_prod']
                beta1_prod.mul_(beta1)

                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg.div(1.0 - beta1_prod).mul_(math.pow(beta0, 1.0 - group['dampening']))

                p.add_(exp_avg_hat, alpha=-group['lr'])

        return loss
