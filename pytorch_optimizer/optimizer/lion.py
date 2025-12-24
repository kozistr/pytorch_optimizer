from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.foreach_utils import (
    foreach_add_,
    foreach_mul_,
)
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient


class Lion(BaseOptimizer):
    """Symbolic Discovery of Optimization Algorithms.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-4,
        betas: Betas = (0.9, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lion'

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
                state['exp_avg'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _can_use_foreach(self, group: ParamGroup) -> bool:  # noqa: PLR0911
        """Check if foreach can be used for this group.

        Foreach is disabled when using features that require per-parameter handling:
        - Complex tensors (view_as_real)
        - Gradient centralization
        - AdaNorm
        - Cautious updates
        - Maximize
        """
        if group.get('foreach') is False:
            return False

        if self.maximize:
            return False

        if group.get('use_gc') or group.get('adanorm') or group.get('cautious'):
            return False

        params = [p for p in group['params'] if p.grad is not None]
        if len(params) == 0:
            return False

        if any(torch.is_complex(p) for p in params):
            return False

        if any(p.grad.is_sparse for p in params):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
    ) -> None:
        """Foreach-optimized step for a parameter group."""
        beta1, beta2 = group['betas']
        lr = group['lr']

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
            foreach=True,
        )

        updates = [exp_avg.clone() for exp_avg in exp_avgs]
        foreach_mul_(updates, beta1, foreach=True)
        foreach_add_(updates, grads, alpha=1.0 - beta1, foreach=True)

        for u in updates:
            u.sign_()

        foreach_mul_(exp_avgs, beta2, foreach=True)
        foreach_add_(exp_avgs, grads, alpha=1.0 - beta2, foreach=True)

        foreach_add_(params, updates, alpha=-lr, foreach=True)

    def _step_per_param(self, group: ParamGroup) -> None:
        """Per-parameter step (original implementation)."""
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            exp_avg = state['exp_avg']

            p, grad, exp_avg = self.view_as_real(p, grad, exp_avg)

            if group.get('use_gc'):
                centralize_gradient(grad, gc_conv_only=False)

            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            s_grad = self.get_adanorm_gradient(
                grad=grad,
                adanorm=group.get('adanorm', False),
                exp_grad_norm=state.get('exp_grad_adanorm', None),
                r=group.get('adanorm_r', None),
            )

            update = exp_avg.clone()

            update.mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
            exp_avg.mul_(beta2).add_(s_grad, alpha=1.0 - beta2)

            if group.get('cautious'):
                self.apply_cautious(update, grad)

            p.add_(update, alpha=-group['lr'])

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
