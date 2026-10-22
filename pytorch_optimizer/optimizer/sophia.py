
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, HutchinsonG, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class SophiaH(BaseOptimizer):
    """Clipped second-order updates using Hutchinson Hessian estimates.

    Use `loss.backward(create_graph=True)` for internal Hessian estimation, or supply
    external estimates through `step(hessian=...)`.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for gradient momentum and Hessian diagonal estimates.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        p: Maximum absolute entry of the preconditioned update.
        update_period: Number of steps after which to apply Hessian approximation.
        num_samples: Number of noise samples for each Hessian diagonal estimate.
        hessian_distribution: Type of distribution to initialize Hessian.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 6e-2,
        betas: Betas = (0.96, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        p: float = 1e-2,
        update_period: int = 10,
        num_samples: int = 1,
        hessian_distribution: HutchinsonG = 'gaussian',
        eps: float = 1e-12,
        maximize: bool = False,
        foreach: bool | None = None,
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
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
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

            if 'momentum' not in state:
                state['momentum'] = torch.zeros_like(grad)
                state['hessian_moment'] = torch.zeros_like(grad)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        beta1, beta2 = group['betas']
        if self.maximize:
            torch._foreach_neg_(grads)
        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )
        torch._foreach_lerp_(state['momentum'], grads, weight=1.0 - beta1)
        if 'hessian' in state:
            torch._foreach_lerp_(state['hessian_moment'], state['hessian'], weight=1.0 - beta2)
        updates = torch._foreach_clamp_min(state['hessian_moment'], group['eps'])
        updates = torch._foreach_div(state['momentum'], updates)
        torch._foreach_clamp_min_(updates, -group['p'])
        torch._foreach_clamp_max_(updates, group['p'])
        foreach_add_(params, updates, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None, hessian: list[torch.Tensor] | None = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)

        step: int = self.param_groups[0]['step'] + 1
        update_hessian = hessian is not None or (step - 1) % self.update_period == 0

        if hessian is not None:
            self.set_hessian(self.param_groups, self.state, hessian)
        elif update_hessian:
            self.zero_hessian(self.param_groups, self.state)
            self.compute_hutchinson_hessian(
                param_groups=self.param_groups,
                state=self.state,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )

        for group in self.param_groups:
            group['step'] += 1

            beta1, beta2 = group['betas']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                keys = ['momentum', 'hessian_moment'] + (['hessian'] if update_hessian else [])
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=keys)
                if not update_hessian or len(state['hessian']) == len(params):
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

                momentum, hessian_moment = state['momentum'], state['hessian_moment']
                momentum.lerp_(grad, weight=1.0 - beta1)

                if 'hessian' in state and update_hessian:
                    hessian_moment.lerp_(state['hessian'], weight=1.0 - beta2)

                update = hessian_moment.clamp(min=group['eps'])
                torch.div(momentum, update, out=update)
                update.clamp_(-group['p'], group['p'])

                p.add_(update, alpha=-group['lr'])

        return loss
