import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector


class Fira(BaseOptimizer):
    r"""Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint? Fira with AdamW optimizer.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-6,
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
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Fira'

    @torch.no_grad()
    def init_group(self):
        pass

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

            step_size: float = group['lr'] * bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if 'rank' in group and p.dim() == 2:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            projection_type=group['projection_type'],
                        )

                    grad = state['projector'].project(grad, group['step'])

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                norm_grad = exp_avg / de_nom

                if 'rank' in group and p.dim() == 2:
                    sub_grad = state['projector'].project_back(grad)

                    norm_dim: int = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1

                    scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (torch.norm(grad, dim=norm_dim) + 1e-8)
                    if norm_dim == 1:
                        scaling_factor = scaling_factor.unsqueeze(1)

                    scaling_grad = grad.sub(sub_grad).mul_(scaling_factor)

                    if 'scaling_grad' in state:
                        scaling_grad_norm = torch.norm(scaling_grad)

                        limiter = max(scaling_grad_norm / (state['scaling_grad'] + 1e-8), 1.01) / 1.01
                        scaling_grad.div_(limiter)

                        state['scaling_grad'] = scaling_grad_norm / limiter
                    else:
                        state['scaling_grad'] = torch.norm(scaling_grad)

                    norm_grad = state['projector'].project_back(norm_grad).add_(scaling_grad)

                p.add_(norm_grad, alpha=-step_size)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

        return loss
