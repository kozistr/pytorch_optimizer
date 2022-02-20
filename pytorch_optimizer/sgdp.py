import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.utils import projection


class SGDP(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/clovaai/AdamP
    Example :
        from pytorch_optimizer import SGDP
        ...
        model = YourModel()
        optimizer = SGDP(model.parameters())
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
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ):
        """SGDP optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param momentum: float. momentum factor
        :param dampening: float. dampening for momentum
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param delta: float. threshold that determines whether a set of parameters is scale invariant or not
        :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
            on scale-variant parameters
        :param nesterov: bool. enables nesterov momentum
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.wd_ratio = wd_ratio
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            delta=delta,
            wd_ratio=wd_ratio,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_ratio(self.wd_ratio)
        self.validate_epsilon(self.eps)

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
                    raise RuntimeError('SGDP does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                if group['nesterov']:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

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

                if group['weight_decay'] > 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_ratio / (1.0 - momentum))

                p.add_(d_p, alpha=-group['lr'])

        return loss
