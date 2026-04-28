import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class FlashAdamW(BaseOptimizer):
    """AdamW-compatible optimizer from FlashOptim.

    This implementation follows the FlashOptim AdamW update semantics, including optional fully LR-decoupled weight
    decay via ``decouple_lr``. FlashOptim's CUDA/Triton compression features are intentionally not required here so the
    optimizer remains a regular PyTorch optimizer across supported test environments.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and squared gradient.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Decoupled weight decay coefficient.
        decouple_lr (bool): Scale weight decay by ``lr / initial_lr`` instead of ``lr``.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        decouple_lr: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(weight_decay, 'weight_decay')

        if not isinstance(decouple_lr, bool):
            raise ValueError('decouple_lr must be a boolean')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'decouple_lr': decouple_lr,
            **kwargs,
        }

        super().__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def __str__(self) -> str:
        return 'FlashAdamW'

    @staticmethod
    def get_weight_decay_factor(lr: float, initial_lr: float, weight_decay: float, decouple_lr: bool) -> float:
        if weight_decay == 0.0:
            return 0.0
        if decouple_lr:
            return weight_decay * (lr / initial_lr if initial_lr > 0.0 else 0.0)
        return lr * weight_decay

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0
        if 'initial_lr' not in group:
            group['initial_lr'] = group.get('lr')

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

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']
            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))
            step_size: float = group['lr'] / bias_correction1
            weight_decay_factor: float = self.get_weight_decay_factor(
                lr=group['lr'],
                initial_lr=group['initial_lr'],
                weight_decay=group['weight_decay'],
                decouple_lr=group['decouple_lr'],
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                if weight_decay_factor != 0.0:
                    p.mul_(1.0 - weight_decay_factor)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])
                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
