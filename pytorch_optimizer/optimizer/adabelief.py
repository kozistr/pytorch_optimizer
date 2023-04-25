import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaBelief(Optimizer, BaseOptimizer):
    r"""Adapting Step-sizes by the Belief in Observed Gradients.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param n_sma_threshold: number of SMA threshold (recommended is 5).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param rectify: bool. perform the rectified update similar to RAdam.
    :param degenerated_to_sgd: bool. perform SGD update when variance of gradient is high.
    :param amsgrad: bool. whether to use the AMSBound variant.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adamd_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        n_sma_threshold: int = 5,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = True,
        degenerated_to_sgd: bool = True,
        amsgrad: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adamd_debias: bool = False,
        eps: float = 1e-16,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'amsgrad': amsgrad,
            'adanorm': adanorm,
            'adamd_debias': adamd_debias,
            'eps': eps,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'AdaBelief'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['amsgrad']:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

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
            weight_decay = group['weight_decay']

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group['step'])

            if group['rectify']:
                n_sma_max: float = 2.0 / (1.0 - beta2) - 1.0
                beta2_t: float = beta2 ** group['step']
                n_sma: float = n_sma_max - 2 * group['step'] * beta2_t / (1.0 - beta2_t)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['weight_decay'] * (1.0 if group['fixed_decay'] else group['lr']))
                elif weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                s_grad = grad
                if group['adanorm']:
                    grad_norm = torch.linalg.norm(grad)

                    exp_grad_norm = state['exp_grad_norm']
                    exp_grad_norm.mul_(group['r']).add_(grad_norm, alpha=1.0 - group['r'])

                    if exp_grad_norm > grad_norm:
                        s_grad *= exp_grad_norm / grad_norm

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                grad_residual = s_grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2).add_(self.eps)

                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    de_nom = max_exp_avg_var.add(self.eps).sqrt()
                else:
                    de_nom = exp_avg_var.add(self.eps).sqrt()

                de_nom.div_(bias_correction2_sq).add_(self.eps)

                if not group['rectify']:
                    step_size: float = group['lr'] if group['adamd_debias'] else group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                    continue

                if n_sma >= self.n_sma_threshold:
                    step_size = math.sqrt(
                        (1 - beta2_t)
                        * (n_sma - 4)
                        / (n_sma_max - 4)
                        * (n_sma - 2)
                        / n_sma
                        * n_sma_max
                        / (n_sma_max - 2)
                    )
                elif self.degenerated_to_sgd:
                    step_size = 1.0
                else:
                    step_size = -1

                if not group['adamd_debias']:
                    step_size /= bias_correction1

                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_var.sqrt().add_(self.eps)
                    p.addcdiv_(exp_avg, de_nom, value=-step_size * group['lr'])
                elif step_size > 0:
                    p.add_(exp_avg, alpha=-step_size * group['lr'])

        return loss
