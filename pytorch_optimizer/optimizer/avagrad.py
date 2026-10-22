import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcdiv_, group_tensors_by_device_and_dtype


class AvaGrad(BaseOptimizer):
    """Adaptive updates with a lagged second moment preconditioner.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-1,
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
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'gamma': None,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AvaGrad'

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
        bias_correction2_sq: float | torch.Tensor,
        prev_bias_correction2_sq: float | torch.Tensor,
        step_size: float | torch.Tensor,
        first_step: bool,
    ) -> torch.Tensor:
        beta1, beta2 = group['betas']
        exp_avgs, exp_avg_sqs = state['exp_avg'], state['exp_avg_sq']
        if self.maximize:
            torch._foreach_neg_(grads)
        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )
        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        roots = torch._foreach_sqrt(exp_avg_sqs)
        if not first_step:
            de_noms = torch._foreach_div(roots, prev_bias_correction2_sq)
            torch._foreach_add_(de_noms, group['eps'])
            foreach_addcdiv_(params, exp_avgs, de_noms, value=-step_size)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        torch._foreach_div_(roots, bias_correction2_sq)
        torch._foreach_add_(roots, group['eps'])
        return torch.stack([root.norm(-2).pow(-2) for root in roots]).sum()

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
            prev_bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step'] - 1))

            step_size: float = group['lr']
            if group['step'] > 1:
                step_size: float = self.apply_adam_debias(
                    adam_debias=group.get('adam_debias', False),
                    step_size=group['gamma'] * group['lr'],
                    bias_correction1=bias_correction1,
                )

            squared_norm: float | torch.Tensor = 0.0
            num_params: float = 0.0

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    squared_norm += self._step_foreach(
                        group, batch['params'], batch['grads'], batch,
                        bias_correction2_sq, prev_bias_correction2_sq, step_size, group['step'] == 1,
                    )
                num_params = sum(p.numel() for p in params)
                group['gamma'] = 1.0 / math.sqrt(squared_norm / num_params)
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
                sqrt_exp_avg_sq = exp_avg_sq.sqrt()

                if group['step'] > 1:
                    de_nom = sqrt_exp_avg_sq.div(prev_bias_correction2_sq).add_(group['eps'])

                    p.addcdiv_(exp_avg, de_nom, value=-step_size)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                param_wise_lr = sqrt_exp_avg_sq.div_(bias_correction2_sq).add_(group['eps'])
                squared_norm += param_wise_lr.norm(-2) ** -2
                num_params += param_wise_lr.numel()

            group['gamma'] = 0.0 if num_params == 0.0 else 1.0 / math.sqrt(squared_norm / num_params)

        return loss
