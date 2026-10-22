import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class EXAdam(BaseOptimizer):
    """Adam with adaptive cross moment corrections.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations with additional temporary memory. `None` enables them when supported.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = False,
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
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EXAdam'

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
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        bias_correction1: float | torch.Tensor,
        bias_correction2: float | torch.Tensor,
    ) -> None:
        beta1, beta2 = group['betas']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        d1 = torch._foreach_div(exp_avg_sqs, torch._foreach_add(exp_avg_sqs, group['eps']))
        torch._foreach_mul_(d1, 1.0 - bias_correction2)
        torch._foreach_add_(d1, 1.0)

        d2 = torch._foreach_mul(exp_avgs, exp_avgs)
        denominators = torch._foreach_add(d2, group['eps'])
        torch._foreach_div_(d2, denominators)
        torch._foreach_mul_(d2, 1.0 - bias_correction1)
        torch._foreach_add_(d2, 1.0)

        updates = torch._foreach_div(exp_avgs, bias_correction1)
        torch._foreach_mul_(updates, d1)
        torch._foreach_mul_(d1, torch._foreach_div(grads, bias_correction1))
        torch._foreach_add_(updates, d1)

        torch._foreach_mul_(d2, torch._foreach_div(exp_avg_sqs, bias_correction2))
        torch._foreach_sqrt_(d2)
        torch._foreach_add_(d2, group['eps'])

        torch._foreach_div_(updates, d2)
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

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch['exp_avg'], batch['exp_avg_sq'],
                        bias_correction1, bias_correction2,
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

                d1 = exp_avg_sq.add(group['eps'])
                torch.div(exp_avg_sq, d1, out=d1)
                d1.mul_(1.0 - bias_correction2).add_(1.0)

                exp_avg_p2 = exp_avg.pow(2)
                d2 = exp_avg_p2.add(group['eps'])
                torch.div(exp_avg_p2, d2, out=d2)
                d2.mul_(1.0 - bias_correction1).add_(1.0)

                update = exp_avg.div(bias_correction1).mul_(d1)
                d1.mul_(grad.div(bias_correction1))
                update.add_(d1)

                de_nom = d2.mul_(exp_avg_sq.div(bias_correction2)).sqrt_().add_(group['eps'])

                update.div_(de_nom)
                p.add_(update, alpha=-group['lr'])

        return loss
