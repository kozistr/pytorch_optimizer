import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class AdaGC(BaseOptimizer):
    """Improving Training Stability for Large Language Model Pretraining.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas: Coefficients used for computing running averages of gradient and the squared Hessian trace.
        beta (float): Smoothing coefficient for the exponential moving average (EMA).
        lambda_abs (float): Absolute clipping threshold to prevent unstable updates from gradient explosions.
        lambda_rel (float): Relative clipping threshold to prevent unstable updates from gradient explosions.
        warmup_steps (int): Number of warmup steps.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        beta: float = 0.98,
        lambda_abs: float = 1.0,
        lambda_rel: float = 1.05,
        warmup_steps: int = 100,
        weight_decay: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(beta, 'beta', 0.0, 1.0, '[)')
        self.validate_positive(lambda_abs, 'lambda_abs')
        self.validate_positive(lambda_rel, 'lambda_rel')
        self.validate_non_negative(warmup_steps, 'warmup_steps')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'beta': beta,
            'lambda_abs': lambda_abs,
            'lambda_rel': lambda_rel,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaGC'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)
                state['gamma'] = torch.empty((1,), device=grad.device, dtype=grad.dtype)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq, gamma = state['exp_avg'], state['exp_avg_sq'], state['gamma']

                if group['step'] < group['warmup_steps']:
                    grad_norm = get_global_gradient_norm(self.param_groups).add_(group['eps'])

                    h_t = min(group['lambda_abs'] / grad_norm, 1.0)
                    g_hat = grad.mul(h_t)

                    g_hat_norm = g_hat.norm()

                    gamma.copy_(g_hat_norm if group['step'] == 1 else min(gamma, g_hat_norm))
                else:
                    h_t = min(group['lambda_rel'] * gamma / grad.norm(), 1.0)
                    g_hat = grad.mul(h_t)

                    gamma.mul_(group['beta']).add_(g_hat.norm(), alpha=1.0 - group['beta'])

                exp_avg.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_hat, g_hat, value=1.0 - beta2)

                update = (exp_avg / bias_correction1) / exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                p.add_(update, alpha=-group['lr'])

        return loss
