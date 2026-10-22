import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    foreach_add_,
    foreach_addcmul_,
    group_tensors_by_device_and_dtype,
)


class FAdam(BaseOptimizer):
    """Natural gradient Adam using diagonal empirical Fisher information.

    The adaptive stabilizer is `min(eps, eps_2 * RMS(grad)) ** (2 * p)`.
    Checkpoint loading preserves the saved momentum and Fisher state dtypes.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for natural gradient momentum and diagonal empirical Fisher estimates.
        weight_decay: Weight decay coefficient.
        clip: RMS cap for natural gradients and preconditioned weight decay.
        p: Exponent applied to the Fisher information diagonal.
        eps: Upper bound on the adaptive epsilon before applying the exponent.
        momentum_dtype: Dtype of momentum.
        fim_dtype: Data type of the Fisher information diagonal.
        maximize: Maximize the objective instead of minimizing it.
        eps_2: Gradient RMS multiplier for the adaptive epsilon.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        maximize: bool = False,
        eps_2: float = 0.01,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(clip, 'clip')
        self.validate_positive(p, 'p')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(eps_2, 'eps_2')

        self.momentum_dtype = momentum_dtype
        self.fim_dtype = fim_dtype
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'betas': betas,
            'weight_decay': weight_decay,
            'clip': clip,
            'p': p,
            'eps': eps,
            'eps_2': eps_2,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FAdam'

    def _restore_state_types(self, value, saved_value):
        if isinstance(saved_value, torch.Tensor) and saved_value.is_floating_point():
            return saved_value.to(device=value.device)

        return super()._restore_state_types(value, saved_value)

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
                state['momentum'] = torch.zeros_like(p, dtype=self.momentum_dtype)
                state['fim'] = torch.zeros_like(p, dtype=self.fim_dtype)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        curr_beta2: float | torch.Tensor,
    ) -> None:
        momenta, fims = state['momentum'], state['fim']
        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_mul_(fims, curr_beta2)
        foreach_addcmul_(fims, grads, grads, value=1.0 - curr_beta2)
        eps_powers = []
        for grad, fim in zip(grads, fims):
            eps_dtype = torch.promote_types(fim.dtype, torch.float32)
            rms_grad = grad.square().mean().sqrt_()
            scaled_eps = group['eps_2'] * rms_grad
            eps_powers.append(torch.where(
                (rms_grad > 0) & (scaled_eps < group['eps']),
                scaled_eps.pow(2.0 * group['p']).to(fim.dtype).to(eps_dtype),
                group['eps'] ** (2.0 * group['p']),
            ))
        bases = [base.to(power.dtype) for base, power in zip(torch._foreach_pow(fims, group['p']), eps_powers)]
        torch._foreach_add_(bases, eps_powers)
        bases = [base.to(fim.dtype) for base, fim in zip(bases, fims)]

        updates = torch._foreach_div(grads, bases)
        divisors = [(update.square().mean().sqrt_() / group['clip']).clamp_min_(1) for update in updates]
        torch._foreach_div_(updates, divisors)
        torch._foreach_lerp_(momenta, updates, weight=1.0 - group['betas'][0])

        updates = torch._foreach_div(params, bases)
        divisors = [(update.square().mean().sqrt_() / group['clip']).clamp_min_(1) for update in updates]
        torch._foreach_div_(updates, divisors)
        torch._foreach_mul_(updates, group['weight_decay'])
        torch._foreach_add_(updates, momenta)
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

            curr_beta2: float = self.debias_beta(beta2, group['step'])

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=['momentum', 'fim'])
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, curr_beta2)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                momentum, fim = state['momentum'], state['fim']

                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1.0 - curr_beta2)

                rms_grad = grad.pow(2).mean().sqrt_()
                scaled_eps = group['eps_2'] * rms_grad
                eps_dtype = torch.promote_types(fim.dtype, torch.float32)
                eps_power = torch.where(
                    (rms_grad > 0) & (scaled_eps < group['eps']),
                    scaled_eps.pow(2.0 * group['p']).to(fim.dtype).to(eps_dtype),
                    group['eps'] ** (2.0 * group['p']),
                )

                fim_base = fim.pow(group['p']).to(eps_dtype).add_(eps_power).to(fim.dtype)
                update = grad / fim_base

                rms = update.pow(2).mean().sqrt_()
                divisor = (rms / group['clip']).clamp_min_(1)
                update.div_(divisor)

                momentum.lerp_(update, weight=1.0 - beta1)

                torch.div(p, fim_base, out=update)

                rms = torch.pow(update, 2).mean().sqrt_()
                divisor = (rms / group['clip']).clamp_min_(1)
                update.div_(divisor)

                update.mul_(group['weight_decay']).add_(momentum)

                p.add_(update, alpha=-group['lr'])

        return loss
