import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class SGDP(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        eps: float = 1e-8,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
    ):
        defaults: Dict[str, Any] = dict(
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

    @staticmethod
    def channel_view(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[0], -1)

    @staticmethod
    def layer_view(x: torch.Tensor) -> torch.Tensor:
        return x.view(1, -1)

    @staticmethod
    def cosine_similarity(
        x: torch.Tensor, y: torch.Tensor, eps: float, view_func: Callable
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

            if cosine_sim.max() < delta / math.sqrt(
                view_func(p.data).size()[1]
            ):
                p_n = p.data / view_func(p.data).norm(dim=1).view(
                    expand_size
                ).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(
                    expand_size
                )
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure: Optional[Callable] = None) -> float:
        loss: Optional[float] = None
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
                    p.data.mul_(
                        1
                        - group['lr']
                        * group['weight_decay']
                        * wd_ratio
                        / (1 - momentum)
                    )

                p.data.add_(d_p, alpha=-group['lr'])

        return loss
