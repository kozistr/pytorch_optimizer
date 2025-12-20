import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class FAdam(BaseOptimizer):
    """Adam is a natural gradient optimizer using diagonal empirical Fisher information.

    Args:
        params (Parameters): Parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        clip (float): Maximum norm of the gradient.
        p (float): Momentum factor.
        eps (float): Term added to the denominator to improve numerical stability.
        momentum_dtype (torch.dtype): Dtype of momentum.
        fim_dtype (torch.dtype): Dtype of Fisher information matrix.
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        maximize: bool = False,
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
        self.maximize = maximize

        defaults: Defaults = {
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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['momentum'] = torch.zeros_like(p, dtype=self.momentum_dtype)
                state['fim'] = torch.zeros_like(p, dtype=self.fim_dtype)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            curr_beta2: float = self.debias_beta(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

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
