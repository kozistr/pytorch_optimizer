import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class StableAdamW(BaseOptimizer):
    """Stable and low-precision training for large-scale vision-language models.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        kahan_sum (bool): Enables Kahan summation for more accurate parameter updates when training in low precision
            (float16 or bfloat16).
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Decoupled weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.99),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'kahan_sum': kahan_sum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableAdamW'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
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

                state['kahan_comp'] = (
                    torch.zeros_like(p)
                    if (group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16})
                    else None
                )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step'])
            beta2_hat: float = self.debias_beta(beta2, group['step'])

            eps_p2: float = math.pow(group['eps'], 2)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                exp_avg.lerp_(grad, weight=beta1_comp)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1.0 - beta2_hat)

                lr: float = group['lr'] / self.get_stable_adamw_rms(grad, exp_avg_sq, eps=eps_p2)

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                if group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16}:
                    kahan_comp = state['kahan_comp']
                    kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-lr)

                    grad.copy_(p.detach())
                    p.add_(kahan_comp)

                    kahan_comp.add_(grad.sub_(p))
                else:
                    p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-lr)

        return loss
