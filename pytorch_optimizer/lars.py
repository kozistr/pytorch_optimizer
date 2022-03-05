import torch
from torch.optim import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class LARS(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/facebookresearch/mae/blob/main/util/lars.py
    Example :
        from pytorch_optimizer import LARS
        ...
        model = YourModel()
        optimizer = LARS(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
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
        """LARS optimizer, no rate scaling or weight decay for parameters <= 1D
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param weight_decay: float. weight decay (L2 penalty)
        :param momentum: float. momentum
        :param trust_coefficient: float. trust_coefficient
        :param eps: float. epsilon
        """
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
                    raise RuntimeError('LARS does not support sparse gradients')

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
