import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    foreach_addcdiv_,
    foreach_addcmul_,
    foreach_lerp_,
    group_tensors_by_device_and_dtype,
)


class DualAdam(BaseOptimizer):
    """Adam with a decaying inverse Adam update contribution.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and squared gradient.
        switch_rate: Linear decay rate for the inverse Adam update contribution.
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
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        switch_rate: float = 1e-2,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(switch_rate, 'switch_rate', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'switch_rate': switch_rate,
            'weight_decay': weight_decay,
            'foreach': foreach,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DualAdam'

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
        bias_correction1: float | torch.Tensor,
        bias_correction2: float | torch.Tensor,
        inverse_adam_rate: float | torch.Tensor,
        use_inverse_adam: bool,
    ) -> None:
        beta1, beta2 = group['betas']
        exp_avgs, exp_avg_sqs = state['exp_avg'], state['exp_avg_sq']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        exp_avg_hats = torch._foreach_div(exp_avgs, bias_correction1)
        de_noms = torch._foreach_div(exp_avg_sqs, bias_correction2)
        torch._foreach_sqrt_(de_noms)
        torch._foreach_add_(de_noms, group['eps'])

        if use_inverse_adam:
            updates = torch._foreach_reciprocal(de_noms)
            foreach_lerp_(updates, de_noms, weight=inverse_adam_rate)
            foreach_addcmul_(params, exp_avg_hats, updates, value=-group['lr'])
        else:
            foreach_addcdiv_(params, exp_avg_hats, de_noms, value=-group['lr'])

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
            bias_correction2: float = self.debias(beta2, group['step'])

            inverse_adam_rate: float = max(0.0, 1.0 - group['step'] * group['switch_rate'])
            use_inverse_adam: bool = inverse_adam_rate >= group['switch_rate']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch,
                        bias_correction1, bias_correction2, inverse_adam_rate, use_inverse_adam,
                    )
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
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_hat = exp_avg.div(bias_correction1)
                de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(group['eps'])

                if use_inverse_adam:
                    update = de_nom.reciprocal().lerp_(de_nom, weight=inverse_adam_rate)

                    p.addcmul_(exp_avg_hat, update, value=-group['lr'])
                else:
                    p.addcdiv_(exp_avg_hat, de_nom, value=-group['lr'])

        return loss
