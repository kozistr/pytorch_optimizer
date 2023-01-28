import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaPNM(Optimizer, BaseOptimizer):
    r"""Adam + Positive-Negative Momentum Optimizers

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use weight_decouple.
    :param amsgrad: bool. whether to use the AMSGrad variant of this algorithm from the paper.
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        amsgrad: bool = True,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.amsgrad = amsgrad
        self.adamd_debias_term = adamd_debias_term
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            amsgrad=amsgrad,
            adamd_debias_term=adamd_debias_term,
            eps=eps,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @property
    def __name__(self) -> str:
        return 'AdaPNM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['neg_exp_avg'] = torch.zeros_like(p)
                if group['amsgrad']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            noise_norm = math.sqrt((1 + beta3) ** 2 + beta3 ** 2)  # fmt: skip
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    grad.add_(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['neg_exp_avg'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq = state['exp_avg_sq']
                if state['step'] % 2 == 1:
                    exp_avg, neg_exp_avg = state['exp_avg'], state['neg_exp_avg']
                else:
                    exp_avg, neg_exp_avg = state['neg_exp_avg'], state['exp_avg']

                exp_avg.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)  # fmt: skip
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if group['amsgrad']:
                    torch.max(state['max_exp_avg_sq'], exp_avg_sq, out=exp_avg_sq)

                de_nom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr']
                if not group['adamd_debias_term']:
                    step_size /= bias_correction1

                pn_momentum = exp_avg.mul(1.0 + beta3).add(neg_exp_avg, alpha=-beta3).mul(1.0 / noise_norm)
                p.addcdiv_(pn_momentum, de_nom, value=-step_size)

        return loss
