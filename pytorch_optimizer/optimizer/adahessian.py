from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import HUTCHINSON_G, Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class AdaHessian(BaseOptimizer):
    """An Adaptive Second Order Optimizer for Machine Learning.

    Requires `loss.backward(create_graph=True)` in order to calculate Hessians.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        hessian_power (float): Exponent applied to the Hessian trace for scaling updates.
        update_period (int): Number of steps after which to apply the Hessian approximation.
        num_samples (int): Number of times to sample `z` when approximating the Hessian trace.
        hessian_distribution (HUTCHINSON_G): Type of distribution used to initialize the Hutchinson trace estimator.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        hessian_power: float = 1.0,
        update_period: int = 1,
        num_samples: int = 1,
        hessian_distribution: HUTCHINSON_G = 'rademacher',
        eps: float = 1e-16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_range(hessian_power, 'Hessian Power', 0, 1, range_type='(]')

        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'hessian_power': hessian_power,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaHessian'

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

            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_hessian_diag_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None, hessian: Optional[List[torch.Tensor]] = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step: int = self.param_groups[0].get('step', 1)

        if hessian is not None:
            self.set_hessian(self.param_groups, self.state, hessian)
        elif step % self.update_period == 0:
            self.zero_hessian(self.param_groups, self.state)
            self.compute_hutchinson_hessian(
                param_groups=self.param_groups,
                state=self.state,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = self.apply_adam_debias(group.get('adam_debias', False), group['lr'], bias_correction1)

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

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                if 'hessian' in state and (group['step'] % self.update_period == 0 or hessian is not None):
                    exp_hessian_diag_sq.mul_(beta2).addcmul_(state['hessian'], state['hessian'], value=1.0 - beta2)

                de_nom = (exp_hessian_diag_sq / bias_correction2).pow_(group['hessian_power'] / 2).add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
