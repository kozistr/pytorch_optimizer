import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class LARS(Optimizer, BaseOptimizer):
    r"""Layer-wise Adaptive Rate Scaling (no rate scaling or weight decay for parameters <= 1D)

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param momentum: float. momentum.
    :param dampening: float. dampening for momentum.
    :param trust_coefficient: float. trust_coefficient.
    :param nesterov: bool. enables nesterov momentum.
    :param eps: float. epsilon.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        trust_coefficient: float = 1e-3,
        nesterov: bool = False,
        eps: float = 1e-6,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.trust_coefficient = trust_coefficient
        self.nesterov = nesterov
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            trust_coefficient=trust_coefficient,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_weight_decay(self.weight_decay)
        self.validate_momentum(self.momentum)
        self.validate_trust_coefficient(self.trust_coefficient)
        self.validate_epsilon(self.eps)

    @property
    def __name__(self) -> str:
        return 'Lars'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['mu'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    param_norm = p.norm(2.0)
                    update_norm = grad.norm(2.0)
                    one = torch.ones_like(param_norm, device=param_norm.device)

                    trust_ratio = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0.0, (group['trust_coefficient'] * param_norm / update_norm), one),
                        one,
                    )

                    grad.add_(p, alpha=group['weight_decay'])
                    grad.mul_(trust_ratio)

                if group['momentum'] > 0.0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = grad.clone().detach()

                    mu = param_state['momentum_buffer']
                    mu.mul_(group['momentum']).add_(grad, alpha=1.0 - group['dampening'])

                    if group['nesterov']:
                        grad.add_(mu, alpha=group['momentum'])
                    else:
                        grad.copy_(mu)

                p.add_(grad, alpha=-group['lr'])

        return loss
