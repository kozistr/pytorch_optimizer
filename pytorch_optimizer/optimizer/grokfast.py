import math
from collections import deque
from typing import Dict, Literal, Optional

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

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
    r"""Grokfast-MA.

    Example:
    -------
        Here's an example::

            loss.backwards()  # Calculate the gradients.

            grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)

            optimizer.step()  # Call the optimizer.

    :param model: nn.Module. model that contains every trainable parameters.
    :param grads: Optional[Dict[str, deque]]. running memory (Queue for windowed moving average). initialize by setting
        it to None. feed the output of the method recursively after on.
    :param window_size: int. the width of the filter window. additional memory requirements increases linearly with
        respect to the windows size.
    :param lamb: float. amplifying factor hyperparameter of the filter.
    :param filter_type: FILTER_TYPE. aggregation method for the running queue.
    :param warmup: bool. if true, filter is not applied until the queue is filled.
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
                    raise ValueError(f'not supported filter_type {filter_type}')

                p.grad.add_(avg, alpha=lamb)

    return grads


@torch.no_grad()
def gradfilter_ema(
    model: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    r"""Grokfast.

    Example:
    -------
        Here's an example::

            loss.backwards()  # Calculate the gradients.

            grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)

            optimizer.step()  # Call the optimizer.

    :param model: nn.Module. model that contains every trainable parameters.
    :param grads: Optional[Dict[str, deque]]. running memory (EMA). Initialize by setting it to None. Feed the output
        of the method recursively after on.
    :param alpha: int. momentum hyperparameter of the EMA.
    :param lamb: float. amplifying factor hyperparameter of the filter.
    """
    if grads is None:
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad:
            grads[n].mul_(alpha).add_(p.grad, alpha=1.0 - alpha)
            p.grad.add_(grads[n], alpha=lamb)

    return grads


class GrokFastAdamW(BaseOptimizer):
    r"""Accelerated Grokking by Amplifying Slow Gradients with AdamW.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param grokfast: bool. whether to use grokfast.
    :param grokfast_alpha: float. momentum hyperparameter of the EMA.
    :param grokfast_lamb: float. amplifying factor hyperparameter of the filter.
    :param grokfast_after_step: int. warmup step for grokfast.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-4,
        betas: BETAS = (0.9, 0.99),
        grokfast: bool = True,
        grokfast_alpha: float = 0.98,
        grokfast_lamb: float = 2.0,
        grokfast_after_step: int = 0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        normalize_lr: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(grokfast_alpha, 'grokfast_alpha', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        if grokfast and normalize_lr:
            lr /= 1.0 + grokfast_lamb

        defaults: DEFAULTS = {
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

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

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
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['grokfast'] and group['grokfast_lamb'] > 0.0:
                        state['grok_exp_avg'] = grad.clone()

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if should_grokfast:
                    grok_exp_avg = state['grok_exp_avg']
                    grok_exp_avg.lerp_(grad, weight=1.0 - group['grokfast_alpha'])

                    grad.add_(grok_exp_avg, alpha=group['grokfast_lamb'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).clamp_(min=group['eps'])

                update = exp_avg.div(bias_correction1).div_(de_nom)

                p.add_(update, alpha=-group['lr'])

        return loss
