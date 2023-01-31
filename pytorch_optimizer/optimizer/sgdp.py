import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import projection


class SGDP(Optimizer, BaseOptimizer):
    r"""SGD + Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor.
    :param dampening: float. dampening for momentum.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param weight_decay: float. weight decay (L2 penalty).
    :param delta: float. threshold that determines whether a set of parameters is scale invariant or not.
    :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
        on scale-variant parameters.
    :param nesterov: bool. enables nesterov momentum.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.wd_ratio = wd_ratio
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'dampening': dampening,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_ratio(self.wd_ratio)
        self.validate_epsilon(self.eps)

    @property
    def __str__(self) -> str:
        return 'SGDP'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['momentum'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__str__)

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                d_p = buf
                if group['nesterov']:
                    d_p.mul_(momentum).add_(grad)

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = projection(
                        p,
                        grad,
                        d_p,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                if group['weight_decay'] > 0.0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_ratio / (1.0 - momentum))

                p.add_(d_p, alpha=-group['lr'])

        return loss
