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
    :param trust_coefficient: float. trust_coefficient.
    :param eps: float. epsilon.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        trust_coefficient: float = 0.001,
        eps: float = 1e-6,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.trust_coefficient = trust_coefficient
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
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

        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    grad = grad.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(grad)
                    one = torch.ones_like(param_norm, device=param_norm.device)

                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0.0, (g['trust_coefficient'] * param_norm / update_norm), one),
                        one,
                    )
                    grad = grad.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p, device=p.device)

                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(grad)

                p.add_(mu, alpha=-g['lr'])

        return loss
