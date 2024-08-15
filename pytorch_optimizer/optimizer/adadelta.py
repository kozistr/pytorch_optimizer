import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaDelta(BaseOptimizer):
    r"""An Adaptive Learning Rate Method.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param rho: float. coefficient used for computing a running average of squared gradients.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(rho, 'rho', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'rho': rho,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaDelta'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['square_avg'] = torch.zeros_like(p)
                state['acc_delta'] = torch.zeros_like(p)

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

            rho: float = group['rho']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                square_avg.mul_(rho).addcmul_(grad, grad, value=1.0 - rho)

                std = square_avg.add(group['eps']).sqrt_()
                delta = acc_delta.add(group['eps']).sqrt_().div_(std).mul_(grad)

                acc_delta.mul_(rho).addcmul_(delta, delta, value=1.0 - rho)
                p.add_(delta, alpha=-group['lr'])

        return loss
