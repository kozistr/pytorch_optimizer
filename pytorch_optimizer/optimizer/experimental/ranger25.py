import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.agc import agc


class Ranger25(BaseOptimizer):
    r"""Mixin' every fancy optimizer hacks.

        Here's the components
            * ADOPT
            * AdEMAMix
            * Cautious
            * StableAdamW or Adam-atan2
            * OrthoGrad
            * Adaptive gradient clipping
            * Lookahead

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param alpha: float. usually between 4 and 10 would work well.
    :param t_alpha_beta3: Optional[float]. total number of iterations is preferred when needed.
    :param cautious: bool. whether to use the Cautious variant.
    :param stable_adamw: bool. whether to use stable AdamW variant.
    :param orthograd: bool. whether to use OrthoGrad variant.
    :param eps: Optional[float]. term added to the denominator to improve numerical stability. when eps is None and
        stable_adamw is False, adam-atan2 feature will be used.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.98, 0.9999),
        weight_decay: float = 1e-3,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        cautious: bool = True,
        stable_adamw: bool = True,
        orthograd: bool = True,
        eps: Optional[float] = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_range(lookahead_blending_alpha, 'lookahead_blending_alpha', 0.0, 1.0, '[]')
        self.validate_non_negative(eps, 'eps')

        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.cautious = cautious
        self.stable_adamw: bool = stable_adamw if isinstance(eps, float) else False
        self.orthograd = orthograd

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps if (eps is not None) or (eps is None and not stable_adamw) else 1e-8,
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
                state['slow_momentum'] = p.clone()

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
    def apply_orthogonal_gradients(self, params, eps: float = 1e-16) -> None:
        for p in params:
            if p.grad is None or p.grad.is_sparse:
                continue

            w = p.view(-1)
            g = p.grad.view(-1)

            proj = torch.dot(w, g).div_(torch.dot(w, w).add_(eps))
            g_ortho = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
            g_ortho_scaled = g_ortho.mul_(g.norm(2).div_(g_ortho.norm(2).add_(eps)))

            p.grad.copy_(g_ortho_scaled.view_as(p.grad))

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.orthograd:
            for group in self.param_groups:
                self.apply_orthogonal_gradients(group['params'])

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
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['exp_avg_slow'] = torch.zeros_like(grad)
                    state['slow_momentum'] = p.clone()

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                grad.copy_(agc(p, grad))

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                normed_grad = grad.div(
                    exp_avg_sq.sqrt().clamp_(min=group['eps'] if group['eps'] is not None else 1e-8)
                ).clamp_(-clip, clip)

                exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg_slow.mul_(beta3_t).add_(normed_grad, alpha=1.0 - beta3_t)

                update = exp_avg.clone()
                if self.cautious:
                    self.apply_cautious(update, grad)

                if self.stable_adamw:
                    step_size /= self.get_stable_adamw_rms(grad, exp_avg_sq)

                update.add_(exp_avg_slow, alpha=alpha_t)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq)

                if group['eps'] is not None:
                    p.addcdiv_(update, de_nom.add_(group['eps']), value=-step_size)
                else:
                    p.add_(update.atan2_(de_nom), alpha=-step_size)

                if group['step'] % self.lookahead_merge_time == 0:
                    slow_p = state['slow_momentum']
                    slow_p.lerp_(p, weight=self.lookahead_blending_alpha)
                    p.copy_(slow_p)

        return loss
