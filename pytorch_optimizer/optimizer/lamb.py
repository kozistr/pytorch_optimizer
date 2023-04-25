import math
from typing import Union

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Lamb(Optimizer, BaseOptimizer):
    r"""Large Batch Optimization for Deep Learning.

        This Lamb implementation is based on the paper v3, which does not use de-biasing.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param rectify: bool. perform the rectified update similar to RAdam.
    :param degenerated_to_sgd: bool. degenerated to SGD.
    :param n_sma_threshold: int. (recommended is 5).
    :param grad_averaging: bool. whether apply (1 - beta2) to gradient when calculating running averages of gradient.
    :param max_grad_norm: float. max gradient norm to clip.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param adam: bool. always use trust ratio = 1, which turns this into Adam. Useful for comparison purposes.
    :param pre_norm: bool. perform pre-normalization of all gradients.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        rectify: bool = False,
        degenerated_to_sgd: bool = False,
        n_sma_threshold: int = 5,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        pre_norm: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-6,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.degenerated_to_sgd = degenerated_to_sgd
        self.n_sma_threshold = n_sma_threshold
        self.max_grad_norm = max_grad_norm
        self.pre_norm = pre_norm
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'rectify': rectify,
            'grad_averaging': grad_averaging,
            'max_grad_norm': max_grad_norm,
            'adam': adam,
            'adanorm': adanorm,
            'adam_debias': adam_debias,
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
        self.validate_norm(self.max_grad_norm)

    def __str__(self) -> str:
        return 'Lamb'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        device = self.param_groups[0]['params'][0].device

        global_grad_norm = torch.zeros(1, device=device)
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    global_grad_norm.add_(torch.linalg.norm(p.grad).pow(2))

        global_grad_norm.sqrt_()

        return torch.clamp(max_grad_norm / (global_grad_norm + self.eps), max=1.0)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm = 1.0
        if self.pre_norm:
            grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']
            beta3 = 1.0 - beta1 if group['grad_averaging'] else 1.0

            bias_correction1 = 1.0 - beta1 ** group['step']

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

                if self.pre_norm:
                    grad.div_(grad_norm)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                s_grad = grad
                if group['adanorm']:
                    grad_norm = torch.linalg.norm(grad)

                    exp_grad_norm = state['exp_grad_norm']
                    exp_grad_norm.mul_(group['r']).add_(grad_norm, alpha=1.0 - group['r'])

                    if exp_grad_norm > grad_norm:
                        s_grad *= exp_grad_norm / grad_norm

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(s_grad, alpha=beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                if group['rectify']:
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

                    if not group['adam_debias']:
                        step_size /= bias_correction1
                else:
                    step_size = group['lr']

                if group['rectify']:
                    update = p.clone()
                    if n_sma >= self.n_sma_threshold:
                        de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                        update.addcdiv_(exp_avg, de_nom, value=-step_size)
                    else:
                        update.add_(exp_avg, alpha=-step_size)
                else:
                    update = exp_avg / exp_avg_sq.sqrt().add(group['eps'])

                weight_norm = torch.linalg.norm(p).clamp(0, self.clamp)
                p_norm = torch.linalg.norm(update)
                trust_ratio: float = 1.0 if weight_norm == 0 or p_norm == 0 else weight_norm / (p_norm + self.eps)

                state['weight_norm'] = weight_norm
                state['adam_norm'] = p_norm
                state['trust_ratio'] = trust_ratio

                if group['adam']:
                    trust_ratio = 1.0

                if group['rectify']:
                    if n_sma >= self.n_sma_threshold:
                        p.addcdiv_(exp_avg, de_nom, value=-step_size * trust_ratio)
                    else:
                        p.add_(exp_avg, alpha=-step_size * trust_ratio)
                else:
                    p.add_(update, alpha=-step_size * trust_ratio)

        return loss
