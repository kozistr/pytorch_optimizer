import math
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class SGDP(Optimizer):
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
        """
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate.
        :param momentum: float. momentum factor
        :param dampening: float. dampening for momentum
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param delta: float. threshold that determines whether a set of parameters is scale invariant or not
        :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
            on scale-variant parameters
        :param nesterov: bool. enables Nesterov momentum
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps

        self.check_valid_parameters()

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

    def check_valid_parameters(self):
        if self.lr < 0.0:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if self.weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if self.eps < 0.0:
            raise ValueError(f'Invalid eps : {self.eps}')

    @staticmethod
    def channel_view(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[0], -1)

    @staticmethod
    def layer_view(x: torch.Tensor) -> torch.Tensor:
        return x.view(1, -1)

    @staticmethod
    def cosine_similarity(
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float,
        view_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        x = view_func(x)
        y = view_func(y)
        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def projection(
        self,
        p,
        grad,
        perturb: torch.Tensor,
        delta: float,
        wd_ratio: float,
        eps: float,
    ) -> Tuple[torch.Tensor, float]:
        wd: float = 1.0
        expand_size: List[int] = [-1] + [1] * (len(p.shape) - 1)
        for view_func in (self.channel_view, self.layer_view):
            cosine_sim = self.cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size()[1]):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = self.projection(
                        p,
                        grad,
                        d_p,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio / (1 - momentum))

                p.data.add_(d_p, alpha=-group['lr'])

        return loss
