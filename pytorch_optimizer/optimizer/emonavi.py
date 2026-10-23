import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    foreach_addcdiv_,
    foreach_lerp_,
    group_tensors_by_device_and_dtype,
)


def update_ema(state: dict, loss: float | torch.Tensor) -> dict[str, float]:
    """Update the EMA dictionary for the `short`, `medium`, and `long` terms."""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()

    ema = state.setdefault('ema', {})
    ema['short'] = 0.3 * loss + 0.7 * ema.get('short', loss)
    ema['medium'] = 0.05 * loss + 0.95 * ema.get('medium', loss)
    ema['long'] = 0.01 * loss + 0.99 * ema.get('long', loss)

    return ema


def compute_scalar(ema: dict[str, float]) -> float:
    """Compute a bounded relative change between short- and long term loss averages."""
    scale_base_l = max(ema['long'], 1e-5)
    scale_base_m = max(ema['medium'], 1e-5)

    diff = ema['long'] - ema['short']
    diff_l = diff / scale_base_l
    diff_m = diff / scale_base_m

    if abs(diff_l) < 0.05:
        return math.tanh(diff_l)

    return math.tanh(diff_m) if abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l else math.tanh(diff_l)


def get_coef(scalar: float) -> float:
    """Return a damping coefficient from the magnitude of the loss trend scalar."""
    abs_scaler = abs(scalar)
    return 1.0 - abs_scaler if abs_scaler > 0.25 else 1.0


def get_scalar_ratio(scalar: float, use_shadow: bool) -> float:
    """Return the shadow parameter mixing ratio from the loss trend scalar."""
    if not use_shadow:
        return 0.0

    scalar = abs(scalar)
    return 1.0 - scalar if scalar > 0.625 else 0.0


def get_emo_drive(state: dict, loss: float | torch.Tensor, use_shadow: bool) -> tuple[float, float, float]:
    """Compute the update scale, shadow ratio, and trust value from loss trends."""
    ema = update_ema(state, loss)
    scalar = compute_scalar(ema)
    coef = get_coef(scalar)
    ratio = get_scalar_ratio(scalar, use_shadow)

    trust = math.copysign(1.0 - abs(scalar), scalar)

    if 0.25 < abs(scalar) < 0.5:
        emo_drive = (8.0 * abs(trust)) * (1.0 + 0.1 * trust)
    elif abs(scalar) > 0.75:
        emo_drive = coef
    else:
        emo_drive = 1.0

    return emo_drive, ratio, trust


def prepare_emo_foreach(
    group: ParamGroup,
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    shadows: list[torch.Tensor],
    maximize: bool,
    ratio: float | torch.Tensor,
    trust: float | torch.Tensor,
    active: bool,
    update_shadow: bool = False,
) -> None:
    """Apply decay and loss-driven shadow mixing before updating moments."""
    if maximize:
        torch._foreach_neg_(grads)

    BaseOptimizer.apply_weight_decay_foreach(
        params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
    )

    if group['use_shadow']:
        if active:
            torch._foreach_mul_(params, 1.0 - ratio)
            foreach_add_(params, shadows, alpha=trust)

            if update_shadow:
                torch._foreach_lerp_(shadows, params, weight=group['shadow_weight'])
        else:
            foreach_lerp_(shadows, params, weight=0.1 * trust)


class EmoNavi(BaseOptimizer):
    """Adam style updates with loss driven momentum scaling and optional shadow weights.

    Supply a loss closure to `step()` to enable loss driven scaling.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        use_shadow: Blend parameters with a running shadow copy based on loss trends.
        shadow_weight: Interpolation weight for shadow copy updates during a shadow correction.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor updates. `None` enables them for supported groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.use_shadow = use_shadow
        self.maximize = maximize
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoNavi'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

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
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                if group['use_shadow']:
                    state['shadow'] = p.clone()

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        emo_drive: float | torch.Tensor,
        ratio: float | torch.Tensor,
        trust: float | torch.Tensor,
        active: bool,
    ) -> None:
        beta1, beta2 = group['betas']

        prepare_emo_foreach(
            group, params, grads, state.get('shadow', []), self.maximize, ratio, trust, active, update_shadow=True
        )

        torch._foreach_lerp_(state['exp_avg'], grads, weight=1.0 - beta1)
        torch._foreach_mul_(state['exp_avg_sq'], beta2)
        torch._foreach_addcmul_(state['exp_avg_sq'], grads, grads, value=1.0 - beta2)

        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])

        foreach_addcdiv_(params, state['exp_avg'], de_noms, value=-group['lr'] * emo_drive)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            emo_drive, ratio, trust = get_emo_drive(self.state, loss, group['use_shadow'])

            if self.can_use_foreach(group, group.get('foreach')):
                keys = ['exp_avg', 'exp_avg_sq']
                if group['use_shadow']:
                    keys.append('shadow')
                params, grads, state = self.collect_trainable_params(group, self.state, keys)

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._step_foreach
                    if self._compiled_step_foreach is not None and batch['params'][0].dtype not in (
                        torch.float16,
                        torch.bfloat16,
                    ):
                        update = self._compiled_step_foreach

                    update(group, batch['params'], batch['grads'], batch, emo_drive, ratio, abs(trust), ratio > 0.0)

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

                if group['use_shadow']:
                    shadow = state['shadow']

                    if ratio > 0.0:
                        p.mul_(1.0 - ratio).add_(state['shadow'], alpha=abs(trust))
                        shadow.lerp_(p, weight=group['shadow_weight'])
                    else:
                        leap_ratio: float = 0.1 * abs(trust)
                        shadow.lerp_(p, weight=leap_ratio)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-group['lr'] * emo_drive)

        return loss


class EmoLynx(BaseOptimizer):
    """Sign based momentum updates with EmoNavi loss driven scaling.

    Supply a loss closure to `step()` to enable loss driven scaling.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for update interpolation and gradient momentum.
        use_shadow: Blend parameters with a running shadow copy based on loss trends.
        shadow_weight: Interpolation weight for shadow copy updates during a shadow correction.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor updates. `None` enables them for supported groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.99),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoLynx'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

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
                if group['use_shadow']:
                    state['shadow'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        emo_drive: float | torch.Tensor,
        ratio: float | torch.Tensor,
        trust: float | torch.Tensor,
        active: bool,
    ) -> None:
        beta1, beta2 = group['betas']

        prepare_emo_foreach(group, params, grads, state.get('shadow', []), self.maximize, ratio, trust, active)

        updates = torch._foreach_mul(grads, 1.0 - beta1)
        torch._foreach_add_(updates, state['exp_avg'], alpha=beta1)
        torch._foreach_sign_(updates)

        # Consume the sign update before changing its momentum input during CUDA compilation.
        foreach_add_(params, updates, alpha=-group['lr'] * emo_drive)

        torch._foreach_lerp_(state['exp_avg'], grads, weight=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            emo_drive, ratio, trust = get_emo_drive(self.state, loss, group['use_shadow'])

            if self.can_use_foreach(group, group.get('foreach')):
                keys = ['exp_avg']
                if group['use_shadow']:
                    keys.append('shadow')
                params, grads, state = self.collect_trainable_params(group, self.state, keys)

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._step_foreach
                    if self._compiled_step_foreach is not None and batch['params'][0].dtype not in (
                        torch.float16,
                        torch.bfloat16,
                    ):
                        update = self._compiled_step_foreach

                    update(group, batch['params'], batch['grads'], batch, emo_drive, ratio, abs(trust), ratio > 0.0)

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

                if group['use_shadow']:
                    shadow = state['shadow']
                    if ratio > 0.0:
                        p.mul_(1.0 - ratio).add_(shadow, alpha=abs(trust))
                    else:
                        leap_ratio = 0.1 * abs(trust)
                        shadow.lerp_(p, weight=leap_ratio)

                exp_avg = state['exp_avg']

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1).sign_()
                exp_avg.lerp_(grad, weight=1.0 - beta2)

                p.add_(blended_grad, alpha=-group['lr'] * emo_drive)

        return loss


class EmoFact(BaseOptimizer):
    """Factored adaptive updates with EmoNavi loss driven scaling.

    Supply a loss closure to `step()` to enable loss driven scaling.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for row/column gradient RMS averages and the vector second moment.
        use_shadow: Blend parameters with a running shadow copy based on loss trends.
        shadow_weight: Interpolation weight for shadow copy updates during a shadow correction.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor updates. `None` enables them for supported groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self._compiled_step_foreach: Callable | None = None

        self.lr = lr

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoFact'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

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
                if group['use_shadow']:
                    state['shadow'] = p.clone()

                shape = p.size()

                if len(shape) >= 2:
                    r_shape = [shape[0]] + [1] * (len(shape) - 1)
                    state['exp_avg_r'] = torch.zeros(r_shape, dtype=p.dtype, device=p.device)

                    c_shape = [1, *list(shape[1:])]
                    state['exp_avg_c'] = torch.zeros(c_shape, dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        emo_drive: float | torch.Tensor,
        ratio: float | torch.Tensor,
        trust: float | torch.Tensor,
        active: bool,
        factored: bool,
    ) -> None:
        beta1, beta2 = group['betas']

        prepare_emo_foreach(group, params, grads, state.get('shadow', []), self.maximize, ratio, trust, active)

        if factored:
            squares = torch._foreach_pow(grads, 2)
            rows = [value.mean(dim=tuple(range(1, value.ndim)), keepdim=True) for value in squares]
            columns = [value.mean(dim=0, keepdim=True) for value in squares]
            torch._foreach_add_(rows, group['eps'])
            torch._foreach_add_(columns, group['eps'])
            torch._foreach_sqrt_(rows)
            torch._foreach_sqrt_(columns)
            torch._foreach_lerp_(state['exp_avg_r'], rows, weight=1.0 - beta1)
            torch._foreach_lerp_(state['exp_avg_c'], columns, weight=1.0 - beta1)

            de_noms = torch._foreach_mul(state['exp_avg_r'], state['exp_avg_c'])
            torch._foreach_sqrt_(de_noms)
            torch._foreach_add_(de_noms, group['eps'])
            updates = torch._foreach_div(grads, de_noms)
        else:
            torch._foreach_lerp_(state['exp_avg'], grads, weight=1.0 - beta1)
            torch._foreach_mul_(state['exp_avg_sq'], beta2)
            torch._foreach_addcmul_(state['exp_avg_sq'], grads, grads, value=1.0 - beta2)

            de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
            torch._foreach_add_(de_noms, group['eps'])
            updates = torch._foreach_div(state['exp_avg'], de_noms)

        torch._foreach_sign_(updates)
        foreach_add_(params, updates, alpha=-group['lr'] * emo_drive)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            emo_drive, ratio, trust = get_emo_drive(self.state, loss, group['use_shadow'])

            if self.can_use_foreach(group, group.get('foreach')):
                for factored in (True, False):
                    update_group = {
                        **group,
                        'params': [p for p in group['params'] if (p.ndim >= 2) == factored],
                    }
                    keys = ['exp_avg_r', 'exp_avg_c'] if factored else ['exp_avg', 'exp_avg_sq']
                    if group['use_shadow']:
                        keys.append('shadow')
                    params, grads, state = self.collect_trainable_params(update_group, self.state, keys)

                    for batch in group_tensors_by_device_and_dtype(params, grads, state):
                        update = self._step_foreach
                        if self._compiled_step_foreach is not None and batch['params'][0].dtype not in (
                            torch.float16,
                            torch.bfloat16,
                        ):
                            update = self._compiled_step_foreach

                        update(
                            group,
                            batch['params'],
                            batch['grads'],
                            batch,
                            emo_drive,
                            ratio,
                            abs(trust),
                            ratio > 0.0,
                            factored,
                        )

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

                if group['use_shadow']:
                    shadow = state['shadow']
                    if ratio > 0.0:
                        p.mul_(1.0 - ratio).add_(shadow, alpha=abs(trust))
                    else:
                        leap_ratio = 0.1 * abs(trust)
                        shadow.lerp_(p, weight=leap_ratio)

                if grad.dim() >= 2:
                    exp_avg_r, exp_avg_c = state['exp_avg_r'], state['exp_avg_c']

                    grad_p2 = grad.pow(2)
                    r_sq = (
                        torch.mean(grad_p2, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps']).sqrt_()
                    )
                    c_sq = torch.mean(grad_p2, dim=0, keepdim=True).add_(group['eps']).sqrt_()

                    exp_avg_r.lerp_(r_sq, weight=1.0 - beta1)
                    exp_avg_c.lerp_(c_sq, weight=1.0 - beta1)

                    de_nom = (exp_avg_r * exp_avg_c).sqrt_().add_(group['eps'])

                    update = grad / de_nom
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    update = exp_avg / de_nom

                update.sign_()

                p.add_(update, alpha=-group['lr'] * emo_drive)

        self.prev_loss = loss

        return loss
