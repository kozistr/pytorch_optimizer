import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class FOCUS(BaseOptimizer):
    r"""First Order Concentrated Updating Scheme.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param gamma: float. control the strength of the attraction.
    :param weight_decay: float. weight decay (L2 penalty).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        betas: BETAS = (0.9, 0.999),
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(gamma, 'gamma', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'gamma': gamma, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FOCUS'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['pbar'] = torch.zeros_like(p)

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

            beta1, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            weight_decay: float = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['pbar'] = torch.zeros_like(p)

                exp_avg, pbar = state['exp_avg'], state['pbar']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                pbar.mul_(beta2).add_(p, alpha=1.0 - beta2)

                pbar_hat = pbar / bias_correction2

                if weight_decay > 0.0:
                    p.add_(pbar_hat, alpha=-group['lr'] * weight_decay)

                update = (p - pbar_hat).sign_().mul_(group['gamma']).add_(torch.sign(exp_avg))

                p.add_(update, alpha=-group['lr'])

        return loss
