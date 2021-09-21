import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class AdamP(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ):
        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps']
                )
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio: float = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self.projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                if group['weight_decay'] > 0:
                    p.data.mul_(
                        1 - group['lr'] * group['weight_decay'] * wd_ratio
                    )

                p.data.add_(perturb, alpha=-step_size)

        return loss
