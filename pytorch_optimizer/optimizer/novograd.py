import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class NovoGrad(BaseOptimizer):
    """Adaptive updates with layer wise squared gradient norms.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        grad_averaging: Scale new gradient contributions by `1 - beta1`.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.95, 0.98),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        grad_averaging: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'grad_averaging': grad_averaging,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'NovoGrad'

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

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        first_update: bool,
        step_size: float | torch.Tensor,
    ) -> None:
        beta1, beta2 = group['betas']
        if self.maximize:
            torch._foreach_neg_(grads)

        if not first_update:
            squares = [grad.pow(2).sum() for grad in grads]
            torch._foreach_lerp_(state['grads_ema'], squares, weight=1.0 - beta2)

        de_noms = torch._foreach_sqrt(state['grads_ema'])
        torch._foreach_add_(de_noms, group['eps'])
        torch._foreach_div_(grads, de_noms)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        moments = state['moments']
        if first_update:
            torch._foreach_copy_(moments, grads)
        else:
            if group['grad_averaging']:
                torch._foreach_mul_(grads, 1.0 - beta1)
            torch._foreach_mul_(moments, beta1)
            torch._foreach_add_(moments, grads)

        foreach_add_(params, moments, alpha=-step_size)

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

            step_size: float = group['lr']
            if group.get('adam_debias', False):
                step_size = step_size * math.sqrt(self.debias(beta2, group['step']))

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                first_updates = [not self.state[p] for p in params]
                for p, grad, first in zip(params, grads, first_updates):
                    if first:
                        self.state[p]['grads_ema'] = grad.pow(2).sum()
                        self.state[p]['moments'] = torch.zeros_like(p)
                for first in (True, False):
                    selected = [p for p, initial in zip(params, first_updates) if initial == first]
                    selected_grads = [p.grad for p in selected]
                    state = {key: [self.state[p][key] for p in selected] for key in ['grads_ema', 'moments']}
                    for batch in group_tensors_by_device_and_dtype(selected, selected_grads, state):
                        self._step_foreach(group, batch['params'], batch['grads'], batch, first, step_size)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]
                first_update = len(state) == 0
                grad_p2 = grad.pow(2).sum()

                if first_update:
                    state['grads_ema'] = grad_p2
                else:
                    state['grads_ema'].lerp_(grad_p2, weight=1.0 - beta2)

                de_nom = state['grads_ema'].sqrt().add_(group['eps'])
                grad.div_(de_nom)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if first_update:
                    state['moments'] = grad.clone()
                else:
                    if group['grad_averaging']:
                        grad.mul_(1.0 - beta1)

                    state['moments'].mul_(beta1).add_(grad)

                p.add_(state['moments'], alpha=-step_size)

        return loss
