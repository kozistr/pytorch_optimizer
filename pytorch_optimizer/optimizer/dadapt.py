# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import to_real


class DAdaptAdaGrad(Optimizer, BaseOptimizer):
    r"""AdaGrad with D-Adaptation. Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
        Default is inf, for unrestricted.
    :param weight_decay: float. weight decay (L2 penalty).
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        momentum: float = 0.0,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        eps: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.d0 = d0
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'gsq_weighted': 0.0,
            'sk_sq_weighted': 0.0,
            'sk_l1': 0.0,
            'k': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'DAdaptAdaGrad'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['alpha_k'] = torch.full_like(p, fill_value=1e-6)
                state['sk'] = torch.zeros_like(p)
                state['x0'] = torch.clone(p)
                if p.grad.is_sparse:
                    state['weighted_sk'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        lr, momentum = group['lr'], group['momentum']

        growth_rate = group['growth_rate']
        gsq_weighted, sk_sq_weighted = group['gsq_weighted'], group['sk_sq_weighted']
        sk_l1 = group['sk_l1']

        d = group['d']
        d_lr = float(d * lr)

        g_sq = torch.tensor([0.0], device=group['params'][0].device)
        sk_sq_weighted_change = torch.tensor([0.0], device=group['params'][0].device)
        sk_l1_change = torch.tensor([0.0], device=group['params'][0].device)

        for group in self.param_groups:
            weight_decay, eps = group['weight_decay'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if 'alpha_k' not in state:
                    state['alpha_k'] = torch.full_like(p, fill_value=1e-6)
                    state['sk'] = torch.zeros_like(p)
                    state['x0'] = torch.clone(p)
                    if grad.is_sparse:
                        state['weighted_sk'] = torch.zeros_like(p)

                sk, alpha_k = state['sk'], state['alpha_k']

                if grad.is_sparse:
                    weighted_sk = state['weighted_sk']

                    grad = grad.coalesce()

                    vk = grad._values().pow(2)
                    sk_masked = sk.sparse_mask(grad).coalesce()
                    old_sk_l1_masked = sk_masked._values().abs().sum()

                    sk.add_(grad, alpha=d_lr)

                    sk_masked = sk.sparse_mask(grad).coalesce()
                    alpha_k_masked = alpha_k.sparse_mask(grad).coalesce()
                    weighted_sk_masked = weighted_sk.sparse_mask(grad).coalesce()

                    # update alpha before step
                    alpha_k_p1_masked = alpha_k_masked._values() + vk

                    alpha_k_delta_masked = alpha_k_p1_masked - alpha_k_masked._values()
                    alpha_k_delta = torch.sparse_coo_tensor(grad.indices(), alpha_k_delta_masked, grad.shape)
                    alpha_k.add_(alpha_k_delta)

                    de_nom = torch.sqrt(alpha_k_p1_masked + eps)

                    grad_sq = vk.div(de_nom).sum()
                    g_sq.add_(grad_sq)

                    # update weighted sk sq tracking
                    weighted_sk_p1_masked = sk_masked._values().pow(2).div(de_nom)

                    sk_sq_weighted_change.add_(weighted_sk_p1_masked.sum() - weighted_sk_masked._values().sum())

                    weighted_sk_p1_delta_masked = weighted_sk_p1_masked - weighted_sk_masked._values()
                    weighted_sk_p1_delta = torch.sparse_coo_tensor(
                        grad.indices(), weighted_sk_p1_delta_masked, grad.shape
                    )
                    weighted_sk.add_(weighted_sk_p1_delta)

                    sk_l1_masked = sk_masked._values().abs().sum()
                    sk_l1_change.add_(sk_l1_masked - old_sk_l1_masked)
                else:
                    if weight_decay > 0.0:
                        grad.add_(p, alpha=weight_decay)

                    old_sk_sq_weighted_param = sk.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    old_sk_l1_param = sk.abs().sum()

                    alpha_k.add_(grad.pow(2))
                    grad_sq = grad.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    g_sq.add_(grad_sq)

                    sk.add_(grad, alpha=d_lr)

                    sk_sq_weighted_param = sk.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    sk_l1_param = sk.abs().sum()

                    sk_sq_weighted_change.add_(sk_sq_weighted_param - old_sk_sq_weighted_param)
                    sk_l1_change.add_(sk_l1_param - old_sk_l1_param)

        sk_sq_weighted += sk_sq_weighted_change
        gsq_weighted += d_lr * d_lr * g_sq
        sk_l1 += sk_l1_change

        if lr > 0.0:
            d_hat = (sk_sq_weighted - gsq_weighted) / sk_l1
            d = self.d0 = max(d, min(d_hat, d * growth_rate))

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['sk_sq_weighted'] = sk_sq_weighted
            group['sk_l1'] = sk_l1
            group['d'] = d

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                alpha_k = state['alpha_k']
                sk = state['sk']
                x0 = state['x0']

                if grad.is_sparse:
                    grad = grad.coalesce()

                    sk_masked = sk.sparse_mask(grad).coalesce()._values()
                    alpha_k_masked = alpha_k.sparse_mask(grad).coalesce()._values()
                    x0_masked = x0.sparse_mask(grad).coalesce()._values()
                    p_masked = p.sparse_mask(grad).coalesce()._values()

                    loc_masked = x0_masked - sk_masked.div(torch.sqrt(alpha_k_masked + group['eps']))

                    loc_delta_masked = loc_masked - p_masked
                    loc_delta = torch.sparse_coo_tensor(grad.indices(), loc_delta_masked, grad.shape)
                    p.add_(loc_delta)
                else:
                    z = x0 - sk.div(torch.sqrt(alpha_k) + group['eps'])

                    if momentum > 0.0:
                        p.mul_(momentum).add_(z, alpha=1.0 - momentum)
                    else:
                        p.copy_(z)

            group['k'] += 1

        return loss


class DAdaptAdam(Optimizer, BaseOptimizer):
    r"""Adam with D-Adaptation. Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. betas.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
        Default is inf, for unrestricted.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use AdamW style weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        betas: BETAS = (0.9, 0.999),
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.d0 = d0
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'gsq_weighted': 0.0,
            'k': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'DAdaptAdam'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['step'] = 0
                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        beta1, beta2 = group['betas']
        gsq_weighted, growth_rate = group['gsq_weighted'], group['growth_rate']

        d, lr = group['d'], group['lr']
        d_lr = float(d * lr)

        g_sq = torch.tensor([0.0], device=group['params'][0].device)
        sk_sq_weighted = torch.tensor([0.0], device=group['params'][0].device)
        sk_l1 = torch.tensor([0.0], device=group['params'][0].device)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                grad_power = to_real(grad * grad.conj())

                exp_avg.mul_(beta1).add_(grad, alpha=d_lr * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_power, alpha=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                g_sq.add_(grad_power.div_(de_nom).sum())

                s = state['s']
                s.mul_(beta2).add_(grad, alpha=d_lr * (1.0 - beta2))

                sk_sq_weighted.add_(to_real(s * s.conj()).div_(de_nom).sum())
                sk_l1.add_(s.abs().sum())

        gsq_weighted = beta2 * gsq_weighted + g_sq * (d_lr**2) * (1 - beta2)
        if lr > 0.0:
            d_hat = (sk_sq_weighted / (1.0 - beta2) - gsq_weighted) / sk_l1
            d = max(d, min(d_hat, d * growth_rate))

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                de_nom = de_nom.type(p.type())

                if group['weight_decay'] > 0.0 and group['weight_decouple']:
                    p.add_(p, alpha=-group['weight_decay'] * d_lr)

                p.addcdiv_(exp_avg, de_nom, value=-1)

            group['k'] += 1

        return loss


class DAdaptSGD(Optimizer, BaseOptimizer):
    r"""SGD with D-Adaptation. Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
        Default is inf, for unrestricted.
    :param weight_decay: float. weight decay (L2 penalty).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        momentum: float = 0.0,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.d0 = d0
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'gsq_weighted': 0.0,
            'k': 0,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_weight_decay(self.weight_decay)

    def __str__(self) -> str:
        return 'DAdaptSGD'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['step'] = 0
                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        gsq_weighted, growth_rate = group['gsq_weighted'], group['growth_rate']

        g_sq = torch.tensor([0.0], device=group['params'][0].device)
        sk_sq = torch.tensor([0.0], device=group['params'][0].device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p)
                    state['s'] = torch.zeros_like(p)
                    state['x0'] = torch.clone(p)

                g_sq.add_(grad.pow(2).sum())

        group = self.param_groups[0]
        if group['k'] == 0:
            group['g0_norm'] = g_sq.sqrt().item()

        g0_norm = group['g0_norm']
        if g0_norm == 0:
            g0_norm = 1.0

        d, lr = group['d'], group['lr']
        d_lr = float(d * lr) / g0_norm

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                s = state['s']
                s.add_(p.grad, alpha=d_lr)

                sk_sq.add_(s.pow(2).sum())

        group['gsq_weighted'] = gsq_weighted + d_lr * d_lr * g_sq
        gsq_weighted = group['gsq_weighted']

        if lr > 0.0:
            d_hat = (sk_sq - group['gsq_weighted']) / sk_sq.sqrt()
            d = max(d, min(d_hat, d * growth_rate))
            group['d'] = d

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                z = state['z']
                z.copy_(state['x0'] - state['s'])

                p.mul_(momentum).add_(z, alpha=1.0 - momentum)

            group['k'] += 1

        return loss
