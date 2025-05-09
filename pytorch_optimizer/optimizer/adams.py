import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


class AdamS(BaseOptimizer):
    r"""Adam with stable weight decay.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 1e-4,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamS'

    def init_group(self, group: GROUP, **kwargs) -> None:
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

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        exp_avg_sq_hat_sum: float = 0.0

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                param_size += p.numel()

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group['ams_bound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    exp_avg_sq_hat = max_exp_avg_sq
                else:
                    exp_avg_sq_hat = exp_avg_sq

                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_hat_mean: float = math.sqrt(exp_avg_sq_hat_sum / param_size) + self.defaults['eps']

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'],
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=1.0 / exp_avg_sq_hat_mean,
                )

                exp_avg_sq_hat = state['max_exp_avg_sq'] if group['ams_bound'] else state['exp_avg_sq']
                exp_avg_sq_hat.div_(bias_correction2)

                de_nom = exp_avg_sq_hat.sqrt().add_(group['eps'])

                p.addcdiv_(state['exp_avg'], de_nom, value=-step_size)

        return loss
