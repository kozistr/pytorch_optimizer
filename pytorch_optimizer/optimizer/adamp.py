import math
from collections.abc import Callable

import torch
from torch.nn.functional import cosine_similarity

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


def channel_view(x: torch.Tensor) -> torch.Tensor:
    """Flatten a tensor to one row per output channel."""
    return x.view(x.size()[0], -1)


def layer_view(x: torch.Tensor) -> torch.Tensor:
    """Flatten a tensor to a single row."""
    return x.view(1, -1)


def cosine_similarity_by_view(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    view_func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Calculate cosine similarity by the view.

    Args:
        x: Source tensor.
        y: Destination tensor.
        eps: Small constant epsilon added for numerical stability.
        view_func: Function defining the view (e.g., per channel or per layer).

    """
    x = view_func(x)
    y = view_func(y)
    return cosine_similarity(x, y, dim=1, eps=eps).abs_()


def projection(
    p: torch.Tensor,
    grad: torch.Tensor,
    perturb: torch.Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> tuple[torch.Tensor, float]:
    """Project to remove the radial component from the update vector."""
    wd: float = 1.0
    expand_size: list[int] = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (channel_view, layer_view):
        cosine_sim = cosine_similarity_by_view(grad, p, eps, view_func)

        if cosine_sim.max() < delta / math.sqrt(view_func(p).size()[1]):
            p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
            radial = view_func(p_n * perturb).sum(dim=1).view(expand_size)
            perturb.sub_(p_n.mul_(radial))
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


def projection_foreach(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    updates: list[torch.Tensor],
    group: ParamGroup,
) -> list[torch.Tensor]:
    """Project updates with device-side channel and layer selection."""
    ratios = []

    for p, grad, update in zip(params, grads, updates):
        ratio = p.new_ones(())

        if p.dim() > 1:
            shape = [-1] + [1] * (p.dim() - 1)
            channel = cosine_similarity_by_view(grad, p, group['eps'], channel_view).max() < (
                group['delta'] / math.sqrt(channel_view(p).size(1))
            )
            layer = cosine_similarity_by_view(grad, p, group['eps'], layer_view).max() < (
                group['delta'] / math.sqrt(p.numel())
            )

            norm = torch.where(
                channel, channel_view(p).norm(dim=1).view(shape), p.norm().reshape([1] * p.dim())
            ).add_(group['eps'])
            p_n = p / norm
            radial = torch.where(
                channel,
                channel_view(p_n * update).sum(dim=1).view(shape),
                (p_n * update).sum(),
            )

            update.sub_(torch.where(channel | layer, p_n * radial, 0.0))
            ratio = torch.where(channel | layer, group['wd_ratio'], ratio)

        ratios.append(ratio)

    return ratios


def apply_projection_weight_decay(
    params: list[torch.Tensor], grads: list[torch.Tensor], ratios: list[torch.Tensor], group: ParamGroup
) -> None:
    """Apply weight decay after projected updates have been formed."""
    if group['weight_decay'] == 0.0:
        return

    if group['weight_decouple']:
        decay = group['weight_decay'] * (1.0 if group['fixed_decay'] else group['lr'])
        factors = [1.0 - decay * ratio for ratio in ratios]
        torch._foreach_mul_(params, factors)
    else:
        torch._foreach_add_(grads, params, alpha=group['weight_decay'])


class SGDP(BaseOptimizer):
    """SGD with projected updates for scale invariant weights.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        dampening: Dampening factor for momentum.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        delta: Threshold that determines whether a set of parameters is scale invariant or not.
        wd_ratio: Relative weight decay applied on scale invariant parameters compared to that applied on
            scale-variant parameters.
        nesterov: Use Nesterov momentum.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them when supported. Defaults to `False`.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'momentum': momentum,
            'dampening': dampening,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDP'

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
                state['momentum'] = torch.zeros_like(grad)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        buffers: list[torch.Tensor],
    ) -> None:
        momentum = group['momentum']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_mul_(buffers, momentum)
        torch._foreach_add_(buffers, grads, alpha=1.0 - group['dampening'])

        if group['nesterov']:
            updates = torch._foreach_mul(buffers, momentum)
            torch._foreach_add_(updates, grads)
        else:
            updates = [buffer.clone() for buffer in buffers]

        ratios = projection_foreach(params, grads, updates, group)
        apply_projection_weight_decay(params, grads, [ratio / (1.0 - momentum) for ratio in ratios], group)
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

            momentum = group['momentum']

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(group, self.state, ['momentum'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(group, batch['params'], batch['grads'], batch['momentum'])

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                buf = state['momentum']

                p, grad, buf = self.view_as_real(p, grad, buf)

                buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                d_p = buf.mul(momentum).add_(grad) if group['nesterov'] else buf.clone()

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = projection(
                        p,
                        grad,
                        d_p,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=wd_ratio / (1.0 - momentum),
                )

                p.add_(d_p, alpha=-group['lr'])

        return loss


class AdamP(BaseOptimizer):
    """Adam with projected updates for scale invariant weights.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        delta: Threshold that determines whether a set of parameters is scale invariant or not.
        wd_ratio: Relative weight decay applied on scale invariant parameters compared to that applied on
            scale-variant parameters.
        nesterov: Use Nesterov momentum.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them when supported. Defaults to `False`.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamP'

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
                state['exp_avg_sq'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        step_size: float | torch.Tensor,
        bias_correction2_sq: float,
    ) -> None:
        beta1, beta2 = group['betas']

        if self.maximize:
            torch._foreach_neg_(grads)

        if group.get('use_gc'):
            for grad in grads:
                centralize_gradient(grad, gc_conv_only=False)

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        inv_de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(inv_de_noms, group['eps'])
        torch._foreach_reciprocal_(inv_de_noms)
        torch._foreach_mul_(inv_de_noms, bias_correction2_sq)

        if group.get('cautious'):
            updates = [exp_avg.clone() for exp_avg in exp_avgs]
            for update, grad in zip(updates, grads):
                self.apply_cautious(update, grad)

            if group['nesterov']:
                torch._foreach_lerp_(updates, grads, weight=1.0 - beta1)

            torch._foreach_mul_(updates, inv_de_noms)
        elif group['nesterov']:
            updates = torch._foreach_lerp(exp_avgs, grads, weight=1.0 - beta1)
            torch._foreach_mul_(updates, inv_de_noms)
        else:
            updates = torch._foreach_mul(inv_de_noms, exp_avgs)

        ratios = projection_foreach(params, grads, updates, group)
        apply_projection_weight_decay(params, grads, ratios, group)
        foreach_add_(params, updates, alpha=-step_size)

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'],
                bias_correction1=bias_correction1,
            )

            if not group.get('adanorm') and self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, ['exp_avg', 'exp_avg_sq']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch['exp_avg'], batch['exp_avg_sq'],
                        step_size, bias_correction2_sq,
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                if group.get('use_gc'):
                    centralize_gradient(grad, gc_conv_only=False)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg.lerp_(s_grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                inv_de_nom = exp_avg_sq.sqrt().add_(group['eps']).reciprocal_().mul_(bias_correction2_sq)

                if group.get('cautious'):
                    perturb = exp_avg.clone()
                    self.apply_cautious(perturb, grad)
                    if group['nesterov']:
                        perturb.lerp_(grad, weight=1.0 - beta1)
                    perturb.mul_(inv_de_nom)
                elif group['nesterov']:
                    perturb = exp_avg.lerp(grad, weight=1.0 - beta1).mul_(inv_de_nom)
                else:
                    perturb = inv_de_nom.mul_(exp_avg)

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=wd_ratio,
                )

                p.add_(perturb, alpha=-step_size)

        return loss
