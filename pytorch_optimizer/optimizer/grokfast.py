import math
from collections import deque
from typing import Literal

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT

FILTER_TYPE = Literal['mean', 'sum']


@torch.no_grad()
def gradfilter_ma(
    model: nn.Module,
    grads: dict[str, deque] | None = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: FILTER_TYPE = 'mean',
    warmup: bool = True,
) -> dict[str, deque]:
    """Amplify slow gradient components with a windowed moving average filter.

    Args:
        model: Model whose gradients to modify in place after backward.
        grads: Per parameter gradient queues from the previous call. `None` creates the queues.
        window_size: Number of gradients to retain per parameter.
        lamb: Amplification factor for the filtered gradients.
        filter_type: Queue reduction: `'mean'` or `'sum'`.
        warmup: Wait until each queue is full before applying the filter.

    Returns:
        dict[str, deque]: Gradient queues to pass into the next call.

    Examples:
        ```python
        grads = None
        for inputs, targets in data:
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            grads = gradfilter_ma(model, grads=grads)
            optimizer.step()
        ```

    """
    if filter_type not in ('mean', 'sum'):
        raise NotImplementedError(f'not supported filter_type {filter_type}')

    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.setdefault(n, deque(maxlen=window_size)).append(p.grad.clone())

            if not warmup or len(grads[n]) == window_size:
                if filter_type == 'mean':
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == 'sum':
                    avg = sum(grads[n])
                p.grad.add_(avg, alpha=lamb)

    return grads


@torch.no_grad()
def gradfilter_ema(
    model: nn.Module,
    grads: dict[str, torch.Tensor] | None = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Amplify slow gradient components with an exponential moving average filter.

    Args:
        model: Model whose gradients to modify in place after backward.
        grads: Per parameter gradient averages from the previous call. `None` initializes them.
        alpha: Decay rate for the gradient moving average.
        lamb: Amplification factor for the averaged gradients.

    Returns:
        dict[str, torch.Tensor]: Gradient averages to pass into the next call.

    Examples:
        ```python
        grads = None
        for inputs, targets in data:
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            grads = gradfilter_ema(model, grads=grads)
            optimizer.step()
        ```

    """
    if grads is None:
        grads = {}

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if n not in grads:
                grads[n] = p.grad.clone()
            else:
                grads[n].lerp_(p.grad, weight=1.0 - alpha)
            p.grad.add_(grads[n], alpha=lamb)

    return grads


class GrokFastAdamW(BaseOptimizer):
    """AdamW with amplification of slow gradient components.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        grokfast: Whether to use grokfast.
        grokfast_alpha: Momentum hyperparameter of the EMA.
        grokfast_lamb: Amplifying factor hyperparameter of the filter.
        grokfast_after_step: Warmup step for grokfast.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        normalize_lr: Divide the learning rate by `1 + grokfast_lamb` when Grokfast is enabled.
        eps: Term added to the denominator to improve numerical stability.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Betas = (0.9, 0.99),
        grokfast: bool = True,
        grokfast_alpha: float = 0.98,
        grokfast_lamb: float = 2.0,
        grokfast_after_step: int = 0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        normalize_lr: bool = True,
        eps: float = 1e-8,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(grokfast_alpha, 'grokfast_alpha', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.foreach = foreach
        self.maximize = maximize

        if grokfast and normalize_lr:
            lr /= 1.0 + grokfast_lamb

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'grokfast': grokfast,
            'grokfast_alpha': grokfast_alpha,
            'grokfast_lamb': grokfast_lamb,
            'grokfast_after_step': grokfast_after_step,
            'foreach': foreach,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'GrokFastAdamW'

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
                if group['grokfast'] and group['grokfast_lamb'] > 0.0:
                    grok_exp_avg = grad.clone()
                    self.maximize_gradient(grok_exp_avg, maximize=self.maximize)
                    state['grok_exp_avg'] = grok_exp_avg

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        grok_exp_avgs: list[torch.Tensor],
        should_grokfast: bool,
    ) -> None:
        beta1, beta2 = group['betas']

        bias_correction1: float = self.debias(beta1, group['step'])
        bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        if should_grokfast:
            torch._foreach_lerp_(grok_exp_avgs, grads, weight=1.0 - group['grokfast_alpha'])
            torch._foreach_add_(grads, grok_exp_avgs, alpha=group['grokfast_lamb'])

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(de_noms, bias_correction2_sq)
        torch._foreach_clamp_min_(de_noms, group['eps'])

        updates = torch._foreach_div(exp_avgs, bias_correction1)
        torch._foreach_div_(updates, de_noms)

        torch._foreach_add_(params, updates, alpha=-group['lr'])

    def _step_per_param(self, group: ParamGroup, should_grokfast: bool) -> None:
        beta1, beta2 = group['betas']

        bias_correction1: float = self.debias(beta1, group['step'])
        bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            exp_avg, exp_avg_sq, grok_exp_avg = (
                state['exp_avg'],
                state['exp_avg_sq'],
                state.get('grok_exp_avg', None),
            )

            p, grad, exp_avg, exp_avg_sq, grok_exp_avg = self.view_as_real(p, grad, exp_avg, exp_avg_sq, grok_exp_avg)

            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            if should_grokfast:
                grok_exp_avg.lerp_(grad, weight=1.0 - group['grokfast_alpha'])
                grad.add_(grok_exp_avg, alpha=group['grokfast_lamb'])

            exp_avg.lerp_(grad, weight=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).clamp_(min=group['eps'])

            update = exp_avg.div(bias_correction1).div_(de_nom)

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

            should_grokfast: bool = (
                group['grokfast'] and group['step'] > group['grokfast_after_step'] and group['grokfast_lamb'] > 0.0
            )

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group,
                    self.state,
                    state_keys=['exp_avg', 'exp_avg_sq', 'grok_exp_avg'],
                )
                if params:
                    self._step_foreach(
                        group,
                        params,
                        grads,
                        state_dict['exp_avg'],
                        state_dict['exp_avg_sq'],
                        state_dict['grok_exp_avg'],
                        should_grokfast,
                    )
            else:
                self._step_per_param(group, should_grokfast)

        return loss
