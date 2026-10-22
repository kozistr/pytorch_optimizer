import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class PNM(BaseOptimizer):
    """SGD with alternating positive and negative momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Momentum decay rate and positive negative momentum mixing coefficient.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas, beta_range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'PNM'

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
                state['pos_momentum'] = torch.zeros_like(p)
                state['neg_momentum'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        beta1, beta2 = group['betas']
        noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
        pos_momentums, neg_momentums = state['pos_momentum'], state['neg_momentum']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(pos_momentums, grads, weight=1.0 - beta1 ** 2)
        updates = torch._foreach_mul(pos_momentums, 1.0 + beta2)
        torch._foreach_add_(updates, neg_momentums, alpha=-beta2)
        torch._foreach_mul_(updates, 1.0 / noise_norm)
        foreach_add_(params, updates, alpha=-group['lr'])

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

            beta1_p2: float = beta1 ** 2  # fmt: skip
            noise_norm: float = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)  # fmt: skip

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['pos_momentum', 'neg_momentum']
                )
                if group['step'] % 2 == 0:
                    state['pos_momentum'], state['neg_momentum'] = state['neg_momentum'], state['pos_momentum']
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if group['step'] % 2 == 1:
                    pos_momentum, neg_momentum = state['pos_momentum'], state['neg_momentum']
                else:
                    neg_momentum, pos_momentum = state['pos_momentum'], state['neg_momentum']

                p, grad, pos_momentum, neg_momentum = self.view_as_real(p, grad, pos_momentum, neg_momentum)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                pos_momentum.lerp_(grad, weight=1.0 - beta1_p2)

                delta_p = pos_momentum.mul(1.0 + beta2).add_(neg_momentum, alpha=-beta2).mul_(1.0 / noise_norm)

                p.add_(delta_p, alpha=-group['lr'])

        return loss
