import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class MSVAG(BaseOptimizer):
    r"""Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. Moving average (momentum) constant (scalar tensor or float value).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        beta: float = 0.9,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        defaults: DEFAULTS = {'lr': lr, 'beta': beta}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MSVAG'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['s'] = torch.zeros_like(p)

    @staticmethod
    def get_rho(beta_power: float, beta: float) -> float:
        r"""Get rho."""
        rho: float = (1.0 - beta_power ** 2) * (1.0 - beta) ** 2  # fmt: skip
        rho /= (1.0 - beta) * (1.0 - beta_power) ** 2
        return min(rho, 0.9999)

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

            beta: float = group['beta']
            beta_power: float = beta ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)
                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                m = exp_avg.div(beta_power)
                v = exp_avg_sq.div(beta_power)

                rho: float = self.get_rho(beta_power, beta)

                m_p2 = m.pow(2)
                s = (v - m_p2).div_(1.0 - rho)

                factor = m_p2.div(m_p2 + rho * s)
                torch.nan_to_num(factor, nan=0.0, out=factor)
                factor.clamp_(0.0, 1.0)

                p.add_(m * factor, alpha=-group['lr'])

        return loss
