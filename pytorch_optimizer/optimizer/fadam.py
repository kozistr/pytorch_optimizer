import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class FAdam(BaseOptimizer):
    r"""Adam is a natural gradient optimizer using diagonal empirical Fisher information.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param clip: float. maximum norm of the gradient.
    :param p: float. momentum factor.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param momentum_dtype: torch.dtype. type of momentum.
    :param fim_dtype: torch.dtype. type of fim.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.1,
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(clip, 'clip')
        self.validate_positive(p, 'p')
        self.validate_non_negative(eps, 'eps')

        self.momentum_dtype = momentum_dtype
        self.fim_dtype = fim_dtype

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'clip': clip,
            'p': p,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FAdam'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['momentum'] = torch.zeros_like(p, dtype=self.momentum_dtype)
                state['fim'] = torch.zeros_like(p, dtype=self.fim_dtype)

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

            curr_beta2: float = self.debias_beta(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p, dtype=self.momentum_dtype)
                    state['fim'] = torch.zeros_like(p, dtype=self.fim_dtype)

                momentum, fim = state['momentum'], state['fim']
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1.0 - curr_beta2)

                rms_grad = grad.pow(2).mean().sqrt_()
                curr_eps = min(rms_grad, 1) * group['eps']

                fim_base = fim.pow(group['p']).add_(curr_eps)
                grad_nat = grad / fim_base

                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms) / group['clip']
                grad_nat.div_(divisor)

                momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                grad_weights = p / fim_base

                rms = torch.pow(grad_weights, 2).mean().sqrt_()
                divisor = max(1, rms) / group['clip']
                grad_weights.div_(divisor)

                grad_weights.mul_(group['weight_decay']).add_(momentum)

                p.add_(grad_weights, alpha=-group['lr'])

        return loss
