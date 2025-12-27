from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.foreach_utils import foreach_add_, foreach_lerp_


class Tiger(BaseOptimizer):
    r"""A Tight-fisted Optimizer, an optimizer that is extremely budget-conscious.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        beta (float): Coefficient used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Whether to fix weight decay.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        beta: float = 0.965,
        weight_decay: float = 0.01,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Tiger'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(grad)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
    ) -> None:
        beta = group['beta']
        lr = group['lr']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
            foreach=True,
        )

        foreach_lerp_(exp_avgs, grads, weight=1.0 - beta, foreach=True)

        updates = [exp_avg.sign() for exp_avg in exp_avgs]
        foreach_add_(params, updates, alpha=-lr, foreach=True)

    def _step_per_param(self, group: ParamGroup) -> None:
        beta = group['beta']

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

            exp_avg = state['exp_avg']
            exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)

            p.add_(torch.sign(exp_avg) if not torch.is_complex(exp_avg) else torch.sgn(exp_avg), alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=['exp_avg'])
                if params:
                    self._step_foreach(group, params, grads, state_dict['exp_avg'])
            else:
                self._step_per_param(group)

        return loss
