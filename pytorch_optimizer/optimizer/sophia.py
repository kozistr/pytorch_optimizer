from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import HUTCHINSON_G, Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class SophiaH(BaseOptimizer):
    r"""Second-order Clipped Stochastic Optimization.

    Requires `loss.backward(create_graph=True)` in order to calculate hessians.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        p (float): Clip effective (applied) gradient (p).
        update_period (int): Number of steps after which to apply Hessian approximation.
        num_samples (int): Times to sample z for the approximation of the Hessian trace.
        hessian_distribution: HUTCHINSON_G. Type of distribution to initialize Hessian.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 6e-2,
        betas: Betas = (0.96, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        p: float = 1e-2,
        update_period: int = 10,
        num_samples: int = 1,
        hessian_distribution: HUTCHINSON_G = 'gaussian',
        eps: float = 1e-12,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(p, 'p (gradient clip)')
        self.validate_step(update_period, 'update_period')
        self.validate_positive(num_samples, 'num_samples')
        self.validate_options(hessian_distribution, 'hessian_distribution', ['gaussian', 'rademacher'])
        self.validate_non_negative(eps, 'eps')

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
            'p': p,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SophiaH'

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
                state['momentum'] = torch.zeros_like(grad)
                state['hessian_moment'] = torch.zeros_like(grad)

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

                momentum, hessian_moment = state['momentum'], state['hessian_moment']
                momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                if 'hessian' in state and (group['step'] % self.update_period == 0 or hessian is not None):
                    hessian_moment.mul_(beta2).add_(state['hessian'], alpha=1.0 - beta2)

                update = (momentum / torch.clip(hessian_moment, min=group['eps'])).clamp_(-group['p'], group['p'])

                p.add_(update, alpha=-group['lr'])

        return loss
