import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AvaGrad(BaseOptimizer):
    r"""Domain-independent Dominance of Adaptive Methods.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-1,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-1,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'adam_debias': adam_debias,
            'gamma': None,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AvaGrad'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))
            prev_bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step'] - 1))

            squared_norm: float = 0.0
            num_params: float = 0.0

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

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                exp_avg_sq = state['exp_avg_sq']
                sqrt_exp_avg_sq = exp_avg_sq.sqrt()

                if group['step'] > 1:
                    de_nom = sqrt_exp_avg_sq.div(prev_bias_correction2_sq).add_(group['eps'])

                    step_size: float = self.apply_adam_debias(
                        adam_debias=group['adam_debias'],
                        step_size=group['gamma'] * group['lr'],
                        bias_correction1=bias_correction1,
                    )
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                param_wise_lr = sqrt_exp_avg_sq.div_(bias_correction2_sq).add_(group['eps'])
                squared_norm += param_wise_lr.norm(-2) ** -2
                num_params += param_wise_lr.numel()

            group['gamma'] = 0.0 if num_params == 0.0 else 1.0 / math.sqrt(squared_norm / num_params)

        return loss
