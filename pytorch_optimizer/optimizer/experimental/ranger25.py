import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Ranger25(BaseOptimizer):
    r"""Mixin' every fancy optimizer hacks.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param alpha: float. usually between 4 and 10 would work well.
    :param t_alpha_beta3: Optional[float]. total number of iterations is preferred when needed.
    :param n_sma_threshold: number of SMA threshold (recommended is 5).
    :param cautious: bool. whether to use the Cautious variant.
    :param stable_adamw: bool. whether to use stable AdamW variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999, 0.9999),
        weight_decay: float = 1e-3,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        n_sma_threshold: int = 5,
        cautious: bool = True,
        stable_adamw: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.n_sma_threshold = n_sma_threshold
        self.cautious = cautious
        self.stable_adamw = stable_adamw

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Ranger25'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        return alpha if t_alpha_beta3 is None else min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        log_beta1, log_beta3 = math.log(beta1), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
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

            beta1, beta2, beta3 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] / bias_correction1
            clip: float = math.pow(group['step'], 0.25)

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3)

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
                    state['exp_avg_slow'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                de_nom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                normed_grad = grad.div(de_nom).clamp_(-clip, clip)

                exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg_slow.mul_(beta3_t).add_(normed_grad, alpha=1.0 - beta3_t)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                update = exp_avg.clone()
                if self.cautious:
                    self.apply_cautious(update, grad)

                if self.stable_adamw:
                    step_size /= self.get_stable_adamw_rms(grad, exp_avg_sq)

                step_size, n_sma = self.get_rectify_step_size(
                    is_rectify=True,
                    step=group['step'],
                    lr=step_size,
                    beta2=beta2,
                    n_sma_threshold=self.n_sma_threshold,
                    degenerated_to_sgd=False,
                )

                update.add_(exp_avg_slow, alpha=alpha_t).div_(de_nom)

                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(update, de_nom, value=-step_size)
                else:
                    p.add_(update, alpha=-step_size)

        return loss
