import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import group_tensors_by_device_and_dtype


class Yogi(BaseOptimizer):
    """Adaptive updates with sign controlled second moment accumulation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        initial_accumulator: Initial values for first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Betas = (0.9, 0.999),
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
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

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'initial_accumulator': initial_accumulator,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Yogi'

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
                state['exp_avg'] = torch.full_like(grad, fill_value=group['initial_accumulator'])
                state['exp_avg_sq'] = torch.full_like(grad, fill_value=group['initial_accumulator'])

    def _update_second_moment_foreach(
        self, exp_avg_sqs: list[torch.Tensor], grads: list[torch.Tensor], beta2: float
    ) -> None:
        grad_p2 = torch._foreach_mul(grads, grads)
        signs = torch._foreach_sub(exp_avg_sqs, grad_p2)
        torch._foreach_sign_(signs)
        torch._foreach_addcmul_(exp_avg_sqs, signs, grad_p2, value=-(1.0 - beta2))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        step_size: float,
        bias_correction2_sq: float,
    ) -> None:
        beta1, beta2 = group['betas']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        self._update_second_moment_foreach(exp_avg_sqs, grads, beta2)

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(de_noms, bias_correction2_sq)
        torch._foreach_add_(de_noms, group['eps'])

        torch._foreach_addcdiv_(params, exp_avgs, de_noms, value=-step_size)

    def _step_per_param(self, group: ParamGroup, step_size: float, bias_correction2_sq: float) -> None:
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            grad_p2 = grad.mul(grad)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.lerp_(grad, weight=1.0 - beta1)
            exp_avg_sq.addcmul_(
                (
                    (exp_avg_sq - grad_p2).sign_()
                    if not torch.is_complex(exp_avg_sq)
                    else (exp_avg_sq - grad_p2).sgn_()
                ),
                grad_p2,
                value=-(1.0 - beta2),
            )

            de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

            p.addcdiv_(exp_avg, de_nom, value=-step_size)

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False), step_size=group['lr'], bias_correction1=bias_correction1
            )

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for tensors in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group,
                        tensors['params'],
                        tensors['grads'],
                        tensors['exp_avg'],
                        tensors['exp_avg_sq'],
                        step_size,
                        bias_correction2_sq,
                    )
            else:
                self._step_per_param(group, step_size, bias_correction2_sq)

        return loss
