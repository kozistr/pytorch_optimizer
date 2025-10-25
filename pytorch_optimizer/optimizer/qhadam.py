from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class QHAdam(BaseOptimizer):
    """Quasi-hyperbolic momentum and Adam for deep learning.

    Args:
        params (Parameters): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        betas (Betas): coefficients used for computing running averages of gradient and the squared Hessian trace.
        nus (Tuple[float, float]): immediate discount factors used to estimate the gradient and its square.
        weight_decay (float): weight decay (L2 penalty).
        weight_decouple (bool): the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): fix weight decay.
        eps (float): term added to the denominator to improve numerical stability.
        maximize (bool): maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        nus: Tuple[float, float] = (1.0, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_nus(nus)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'nus': nus,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'QHAdam'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['beta1_weight'] = torch.zeros((1,), dtype=torch.float32, device=grad.device)
                state['beta2_weight'] = torch.zeros((1,), dtype=torch.float32, device=grad.device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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
            nu1, nu2 = group['nus']

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

                beta1_weight, beta2_weight = state['beta1_weight'], state['beta2_weight']
                beta1_weight.mul_(beta1).add_(1.0)
                beta2_weight.mul_(beta2).add_(1.0)

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)

                grad_p2 = grad.pow(2)

                exp_avg.mul_(beta1_adj).add_((1.0 - beta1_adj) * grad)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj * grad_p2)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(grad, alpha=1.0 - nu1)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(grad_p2, alpha=1.0 - nu2)

                avg_grad_rms.sqrt_().add_(group['eps'])

                p.addcdiv_(avg_grad, avg_grad_rms, value=-group['lr'])

        return loss
