import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class AdaDelta(BaseOptimizer):
    """Adaptive updates based on accumulated squared gradients and updates.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        rho: Coefficient used for computing a running average of squared gradients.
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
        lr: float | torch.Tensor = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(rho, 'rho', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'rho': rho,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaDelta'

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
                state['square_avg'] = torch.zeros_like(p)
                state['acc_delta'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        rho = group['rho']
        square_avgs, acc_deltas = state['square_avg'], state['acc_delta']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_mul_(square_avgs, rho)
        torch._foreach_addcmul_(square_avgs, grads, grads, value=1.0 - rho)

        stds = torch._foreach_add(square_avgs, group['eps'])
        torch._foreach_sqrt_(stds)
        deltas = torch._foreach_add(acc_deltas, group['eps'])
        torch._foreach_sqrt_(deltas)
        torch._foreach_div_(deltas, stds)
        torch._foreach_mul_(deltas, grads)

        torch._foreach_mul_(acc_deltas, rho)
        torch._foreach_addcmul_(acc_deltas, deltas, deltas, value=1.0 - rho)
        foreach_add_(params, deltas, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            rho: float = group['rho']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['square_avg', 'acc_delta']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch)
                continue

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

                square_avg, acc_delta = state['square_avg'], state['acc_delta']

                p, grad, square_avg, acc_delta = self.view_as_real(p, grad, square_avg, acc_delta)

                square_avg.mul_(rho).addcmul_(grad, grad, value=1.0 - rho)

                std = square_avg.add(group['eps']).sqrt_()
                delta = acc_delta.add(group['eps']).sqrt_().div_(std).mul_(grad)

                acc_delta.mul_(rho).addcmul_(delta, delta, value=1.0 - rho)

                p.add_(delta, alpha=-group['lr'])

        return loss
