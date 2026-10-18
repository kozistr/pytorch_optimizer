import math
from typing import Literal

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import group_tensors_by_device_and_dtype

VARIANTS = Literal['uni', 'inc', 'exp']


class A2Grad(BaseOptimizer):
    """Adaptive accelerated stochastic gradient descent with three averaging variants.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate. No needed.
        beta: Coefficient controlling the adaptive gradient scale.
        lips: Positive Lipschitz constant.
        rho: Represents the degree of weighting decrease, a constant smoothing factor between 0 and 1.
        variant: Variant of A2Grad optimizer. One of 'uni', 'inc', or 'exp'.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | None = None,
        beta: float = 10.0,
        lips: float = 10.0,
        rho: float = 0.5,
        variant: VARIANTS = 'uni',
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_positive(lips, 'lips')
        self.validate_non_negative(rho, 'rho')
        self.validate_options(variant, 'variant', ['uni', 'inc', 'exp'])

        self.variant = variant
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'beta': beta, 'lips': lips, 'foreach': foreach}
        if variant == 'exp':
            defaults.update({'rho': rho})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'A2Grad'

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
                state['alpha_k'] = 1.0
                state['v_k'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)
                state['avg_grad'] = grad.clone()
                state['x_k'] = p.clone()
                if self.variant == 'exp':
                    state['v_kk'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        gamma_k: float,
        alpha_k_1: float,
    ) -> None:
        avg_grads, x_ks = state_dict['avg_grad'], state_dict['x_k']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_lerp_(avg_grads, grads, weight=1.0 / group['step'])

        deltas = torch._foreach_sub(grads, avg_grads)
        torch._foreach_mul_(deltas, deltas)
        delta_sqs = torch.stack(
            [delta.sum() for delta in deltas]
            if grads[0].dtype in (torch.float16, torch.bfloat16)
            else torch._foreach_norm(deltas, ord=1)
        )

        v_k = torch.cat(state_dict['v_k'])
        if self.variant in ('uni', 'inc'):
            if self.variant == 'inc':
                v_k.mul_((group['step'] / (group['step'] + 1)) ** 2)
            v_k.add_(delta_sqs)
        else:
            v_kk = torch.cat(state_dict['v_kk'])
            v_kk.lerp_(delta_sqs, weight=1.0 - group['rho'])
            torch.maximum(v_k, v_kk, out=v_k)
            torch._foreach_copy_(state_dict['v_kk'], v_kk.split(1))

        torch._foreach_copy_(state_dict['v_k'], v_k.split(1))

        h_k = v_k.sqrt()
        if self.variant != 'uni':
            h_k.mul_(math.sqrt(group['step'] + 1))

        if h_k.dtype in (torch.float16, torch.bfloat16):
            h_k = h_k.float()

        coefficients = h_k.mul_(group['beta']).add_(gamma_k).reciprocal_().neg_().unbind()
        torch._foreach_addcmul_(x_ks, grads, coefficients)
        torch._foreach_lerp_(params, x_ks, weight=alpha_k_1)

        weights = [(1.0 - alpha_k_1) * self.state[p]['alpha_k'] for p in params]
        torch._foreach_mul_(coefficients, weights)
        torch._foreach_addcmul_(params, grads, coefficients)

        for p in params:
            self.state[p]['alpha_k'] = alpha_k_1

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            gamma_k: float = 2.0 * group['lips'] / (group['step'] + 1)
            alpha_k_1: float = 2.0 / (group['step'] + 3)

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                state_keys = ['avg_grad', 'v_k', 'x_k'] + (['v_kk'] if self.variant == 'exp' else [])
                params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=state_keys)
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, gamma_k, alpha_k_1)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                avg_grad, v_k, x_k = state['avg_grad'], state['v_k'], state['x_k']
                avg_grad.lerp_(grad, weight=1.0 / group['step'])

                delta_k = grad.sub(avg_grad)

                delta_k_sq = delta_k.square_().sum()

                if self.variant in ('uni', 'inc'):
                    if self.variant == 'inc':
                        v_k.mul_((group['step'] / (group['step'] + 1)) ** 2)
                    v_k.add_(delta_k_sq)
                else:
                    v_kk = state['v_kk']

                    v_kk.lerp_(delta_k_sq, weight=1.0 - group['rho'])
                    torch.max(v_kk, v_k, out=v_k)

                h_k = v_k.sqrt()
                if self.variant != 'uni':
                    h_k.mul_(math.sqrt(group['step'] + 1))

                coefficient = -1.0 / (gamma_k + group['beta'] * h_k.item())

                x_k.add_(grad, alpha=coefficient)

                p.lerp_(x_k, weight=alpha_k_1)
                p.add_(grad, alpha=(1.0 - alpha_k_1) * state['alpha_k'] * coefficient)

                state['alpha_k'] = alpha_k_1

        return loss
