import math
from collections import deque
from typing import Dict, Literal, Optional, cast

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup

FILTER_TYPE = Literal['mean', 'sum']


@torch.no_grad()
def gradfilter_ma(
    model: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: FILTER_TYPE = 'mean',
    warmup: bool = True,
) -> Dict[str, deque]:
    """Grokfast-MA.

    Args:
        model (nn.Module): Model that contains every trainable parameters.
        grads (Optional[Dict[str, deque]]): Running memory (queue for windowed moving average).
            Initialize by setting  it to None.
            Feed the output of the method recursively after one call.
        window_size (int): The width of the filter window.
            Additional memory requirements increase linearly with window size.
        lamb (float): Amplifying factor hyperparameter of the filter.
        filter_type (FILTER_TYPE): Aggregation method for the running queue.
        warmup (bool): If true, the filter is not applied until the queue is filled.

    Example:
        loss.backwards()  # Calculate the gradients.

        grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)

        optimizer.step()  # Call the optimizer.
    """
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad:
            grads[n].append(p.grad)

            if not warmup or len(grads[n]) == window_size:
                if filter_type == 'mean':
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == 'sum':
                    avg = sum(grads[n])
                else:
                    raise NotImplementedError(f'not supported filter_type {filter_type}')

                p.grad.add_(avg, alpha=lamb)

    return grads


@torch.no_grad()
def gradfilter_ema(
    model: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Grokfast.

    Args:
        model (nn.Module): Model that contains every trainable parameters.
        grads (Optional[Dict[str, deque]]): Running memory (EMA). Initialize by setting it to None.
            Feed the output of the method recursively after one call.
        alpha (int): Momentum hyperparameter of the EMA.
        lamb (float): Amplifying factor hyperparameter of the filter.

    Example:
        loss.backwards()  # Calculate the gradients.

        grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)

        optimizer.step()  # Call the optimizer.
    """
    if grads is None:
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}

    grads = cast(Dict[str, torch.Tensor], grads)

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].mul_(alpha).add_(p.grad, alpha=1.0 - alpha)
            p.grad.add_(grads[n], alpha=lamb)

    return grads


class GrokFastAdamW(BaseOptimizer):
    """Accelerated Grokking by Amplifying Slow Gradients with AdamW.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        grokfast (bool): Whether to use grokfast.
        grokfast_alpha (float): Momentum hyperparameter of the EMA.
        grokfast_lamb (float): Amplifying factor hyperparameter of the filter.
        grokfast_after_step (int): Warmup step for grokfast.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
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
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(grokfast_alpha, 'grokfast_alpha', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

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
                    state['grok_exp_avg'] = grad.clone()

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

            should_grokfast: bool = (
                group['grokfast'] and group['step'] > group['grokfast_after_step'] and group['grokfast_lamb'] > 0.0
            )

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

                p, grad, exp_avg, exp_avg_sq, grok_exp_avg = self.view_as_real(
                    p, grad, exp_avg, exp_avg_sq, grok_exp_avg
                )

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

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).clamp_(min=group['eps'])

                update = exp_avg.div(bias_correction1).div_(de_nom)

                p.add_(update, alpha=-group['lr'])

        return loss
