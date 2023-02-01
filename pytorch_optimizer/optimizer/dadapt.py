# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


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
        lr: float = 1e-3,
        momentum: float = 0.0,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        eps: float = 1e-3,
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
            'sksq_weighted': 0.0,
            'skl1': 0.0,
            'k': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @property
    def __str__(self) -> str:
        return 'DAdaptAdaGrad'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
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

        g_sq = 0.0
        sksq_weighted_change = 0.0
        skl1_change = 0.0

        group = self.param_groups[0]

        lr, momentum = group['lr'], group['momentum']
        ck: float = 1.0 - momentum

        growth_rate = group['growth_rate']
        gsq_weighted, sksq_weighted = group['gsq_weighted'], group['sksq_weighted']
        skl1 = group['skl1']

        d = group['d']
        d_lr = d * lr

        for group in self.param_groups:
            k, weight_decay, eps = group['k'], group['weight_decay'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['alpha_k'] = torch.full_like(p, fill_value=1e-6)
                    state['sk'] = torch.zeros_like(p)
                    state['x0'] = torch.clone(p)
                    if grad.is_sparse:
                        state['weighted_sk'] = torch.zeros_like(p)

                sk, alpha_k = state['sk'], state['alpha_k']

                if grad.is_sparse:
                    weighted_sk = state['weighted_sk']

                    grad = grad.coalesce()
                    grad_vals = grad._values()
                    vk_vals = grad_vals.pow(2)

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()

                    old_skl1_vals = sk_vals.abs().sum()

                    sk.add_(grad, alpha=d_lr)

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()
                    alpha_k_vals = alpha_k.sparse_mask(grad).coalesce()._values()
                    weighted_sk_vals = weighted_sk.sparse_mask(grad).coalesce()._values()

                    # update alpha before step
                    alpha_kp1_vals = alpha_k_vals + vk_vals

                    alpha_k_delta_vals = alpha_kp1_vals - alpha_k_vals
                    alpha_k_delta = torch.sparse_coo_tensor(grad.indices(), alpha_k_delta_vals, grad.shape)
                    alpha_k.add_(alpha_k_delta)

                    de_nom = torch.sqrt(alpha_kp1_vals + eps)

                    grad_sq = grad_vals.pow(2).div(de_nom).sum()
                    g_sq.add_(grad_sq)

                    # update weighted sk sq tracking
                    weighted_skp1_vals = sk_vals.pow(2).div(de_nom)

                    sksq_weighted_change += weighted_skp1_vals.sum() - weighted_sk_vals.sum()

                    weighted_skp1_delta_vals = weighted_skp1_vals - weighted_sk_vals
                    weighted_skp1_delta = torch.sparse_coo_tensor(grad.indices(), weighted_skp1_delta_vals, grad.shape)
                    weighted_sk.add_(weighted_skp1_delta)

                    skl1_vals = sk_vals.abs().sum()

                    skl1_change += skl1_vals - old_skl1_vals
                else:
                    if weight_decay > 0.0:
                        grad.add_(p, alpha=weight_decay)

                    old_sksq_weighted_param = sk.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    old_skl1_param = sk.abs().sum()

                    alpha_k.add_(grad.pow(2))
                    grad_sq = grad.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    g_sq.add_(grad_sq)

                    sk.add_(grad, alpha=d_lr)

                    sksq_weighted_param = sk.pow(2).div(torch.sqrt(alpha_k) + eps).sum()
                    skl1_param = sk.abs().sum()

                    sksq_weighted_change += sksq_weighted_param - old_sksq_weighted_param
                    skl1_change += skl1_param - old_skl1_param

        sksq_weighted += sksq_weighted_change
        skl1 += skl1_change

        gsq_weighted += d_lr * d_lr * g_sq
        d_hat = d

        if lr > 0.0:
            d_hat = (sksq_weighted - gsq_weighted) / skl1
            d = self.d0 = max(d, min(d_hat, d * growth_rate))

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['sksq_weighted'] = sksq_weighted
            group['skl1'] = skl1
            group['d'] = d

            k, weight_decay, eps = group['k'], group['weight_decay'], group['eps']

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

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()
                    alpha_k_vals = alpha_k.sparse_mask(grad).coalesce()._values()
                    x0_vals = x0.sparse_mask(grad).coalesce()._values()
                    p_vals = p.sparse_mask(grad).coalesce()._values()

                    loc_vals = x0_vals - sk_vals.div(torch.sqrt(alpha_k_vals + eps))

                    loc_delta_vals = loc_vals - p_vals
                    loc_delta = torch.sparse_coo_tensor(grad.indices(), loc_delta_vals, grad.shape)
                    p.add_(loc_delta)
                else:
                    z = x0 - sk.div(torch.sqrt(alpha_k) + eps)

                    if momentum > 0.0:
                        p.mul_(1 - ck).add_(z, alpha=ck)
                    else:
                        p.copy_(z)

            group['k'] += 1

        return loss
