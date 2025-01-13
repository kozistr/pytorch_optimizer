import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class StableAdamW(BaseOptimizer):
    r"""Stable and low-precision training for large-scale vision-language models.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param kahan_sum: bool. Enables Kahan summation for more accurate parameter updates when training in low precision
        (float16 or bfloat16).
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.99),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'kahan_sum': kahan_sum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableAdamW'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                state['kahan_comp'] = (
                    torch.zeros_like(p) if group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16} else None
                )

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step'])
            beta2_hat: float = self.debias_beta(beta2, group['step'])

            eps_p2: float = math.pow(group['eps'], 2)

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

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.lerp_(grad, weight=beta1_comp)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1.0 - beta2_hat)

                rms = self.get_stable_adamw_rms(grad, exp_avg_sq, eps=eps_p2)
                lr = group['lr'] / rms

                self.apply_weight_decay(
                    p,
                    grad,
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
