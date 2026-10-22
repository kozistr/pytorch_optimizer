import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class FOCUS(BaseOptimizer):
    """Sign based updates with attraction toward the running parameter average.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for gradient momentum and the running parameter average.
        gamma: Controls the strength of the attraction.
        weight_decay: Weight decay coefficient.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-2,
        betas: Betas = (0.9, 0.999),
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(gamma, 'gamma', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr, 'betas': betas, 'gamma': gamma, 'weight_decay': weight_decay, 'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FOCUS'

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
                state['pbar'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        beta1, beta2 = group['betas']
        exp_avgs, pbars = state['exp_avg'], state['pbar']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_lerp_(pbars, params, weight=1.0 - beta2)
        pbar_hats = torch._foreach_div(pbars, bias_correction2)

        if group['weight_decay'] > 0.0:
            foreach_add_(params, pbar_hats, alpha=-group['lr'] * group['weight_decay'])

        updates = torch._foreach_sub(params, pbar_hats)
        torch._foreach_sign_(updates)
        torch._foreach_mul_(updates, group['gamma'])
        torch._foreach_add_(updates, torch._foreach_sign(exp_avgs))
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

            bias_correction2: float = self.debias(beta2, group['step'])

            weight_decay: float = group['weight_decay']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=['exp_avg', 'pbar'])
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, bias_correction2)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, pbar = state['exp_avg'], state['pbar']

                p, grad, exp_avg, pbar = self.view_as_real(p, grad, exp_avg, pbar)

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                pbar.lerp_(p, weight=1.0 - beta2)

                pbar_hat = pbar / bias_correction2

                if weight_decay > 0.0:
                    p.add_(pbar_hat, alpha=-group['lr'] * weight_decay)

                update = (p - pbar_hat).sign_().mul_(group['gamma']).add_(torch.sign(exp_avg))

                p.add_(update, alpha=-group['lr'])

        return loss
