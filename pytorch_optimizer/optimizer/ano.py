import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    foreach_addcdiv_,
    foreach_lerp_,
    group_tensors_by_device_and_dtype,
)


class Ano(BaseOptimizer):
    """Ano optimizer with adaptive momentum and sign based updates.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and the squared gradient.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        logarithmic_schedule: Enable adaptive beta1 scheduling based on step count.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-4,
        betas: Betas = (0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        logarithmic_schedule: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.logarithmic_schedule = logarithmic_schedule
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'foreach': foreach,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Ano'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        beta1: float | torch.Tensor,
        bias_correction2: float | torch.Tensor,
    ) -> None:
        beta2 = group['betas'][1]
        exp_avgs, exp_avg_sqs = state['exp_avg'], state['exp_avg_sq']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        square_grads = torch._foreach_mul(grads, grads)
        signs = torch._foreach_sub(square_grads, exp_avg_sqs)
        torch._foreach_sign_(signs)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, signs, square_grads, value=1.0 - beta2)

        de_noms = torch._foreach_div(exp_avg_sqs, bias_correction2)
        torch._foreach_sqrt_(de_noms)
        torch._foreach_add_(de_noms, group['eps'])

        updates = torch._foreach_abs(grads)
        torch._foreach_mul_(updates, torch._foreach_sign(exp_avgs))
        foreach_addcdiv_(params, updates, de_noms, value=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            if self.logarithmic_schedule:
                max_t = max(2, group['step'])
                beta1 = 1.0 - 1.0 / math.log(max_t)

            bias_correction2: float = self.debias(beta2, group['step'])

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, beta1, bias_correction2)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg.lerp_(grad, weight=1.0 - beta1)

                square_grad = grad.square()
                sign_term = torch.sign(square_grad - exp_avg_sq)
                exp_avg_sq.mul_(beta2).addcmul_(sign_term, square_grad, value=1.0 - beta2)

                de_nom = square_grad.copy_(exp_avg_sq).div_(bias_correction2).sqrt_().add_(group['eps'])

                p.addcdiv_(grad.abs().mul_(exp_avg.sign()), de_nom, value=-group['lr'])

        return loss
