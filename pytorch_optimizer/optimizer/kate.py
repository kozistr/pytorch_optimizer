import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Kate(Optimizer, BaseOptimizer):
    r"""Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param eta: float. eta.
    :param delta: float. delta. 0.0 or 1e-8.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. epsilon value.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        eta: float = 0.9,
        delta: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(eta, 'eta', 0.0, 1.0, '[)')
        self.validate_range(delta, 'delta', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'eta': eta,
            'delta': delta,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Kate'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['m'] = torch.zeros_like(p)
                state['b'] = torch.zeros_like(p)

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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['b'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                grad_p2 = grad * grad

                m, b = state['m'], state['b']
                b.add_(grad_p2)

                de_nom = b.add(group['eps'])

                m.add_(grad, alpha=group['eta']).add_(grad / de_nom)

                update = m.sqrt()
                update.mul_(grad).div_(de_nom)

                p.add_(update, alpha=-group['lr'])

        return loss
