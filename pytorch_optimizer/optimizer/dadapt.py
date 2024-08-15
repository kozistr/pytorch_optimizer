# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm, to_real


class DAdaptAdaGrad(BaseOptimizer):
    r"""AdaGrad with D-Adaptation. Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
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
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 0.0,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'k': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

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
        device = group['params'][0].device

        d, lr = group['d'], group['lr']
        d_lr: float = d * lr

        g_sq = torch.tensor([0.0], device=device)
        sk_sq_weighted_change = torch.tensor([0.0], device=device)
        sk_l1_change = torch.tensor([0.0], device=device)
        if 'gsq_weighted' not in group:
            group['gsq_weighted'] = torch.tensor([0.0], device=device)
        if 'sk_sq_weighted' not in group:
            group['sk_sq_weighted'] = torch.tensor([0.0], device=device)
        if 'sk_l1' not in group:
            group['sk_l1'] = torch.tensor([0.0], device=device)

        gsq_weighted = group['gsq_weighted']
        sk_sq_weighted = group['sk_sq_weighted']
        sk_l1 = group['sk_l1']

        for group in self.param_groups:
            eps = group['eps']
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
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

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

        sk_sq_weighted.add_(sk_sq_weighted_change)
        gsq_weighted.add_(g_sq, alpha=d_lr ** 2)  # fmt: skip
        sk_l1.add_(sk_l1_change)

        if sk_l1 == 0:
            return loss

        if lr > 0.0:
            d_hat = (sk_sq_weighted - gsq_weighted) / sk_l1
            d = group['d'] = max(d, min(d_hat.item(), d * group['growth_rate']))

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

                alpha_k, sk, x0 = state['alpha_k'], state['sk'], state['x0']

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
                    z = x0 - sk.div(alpha_k.sqrt().add_(group['eps']))

                    if group['momentum'] > 0.0:
                        p.mul_(group['momentum']).add_(z, alpha=1.0 - group['momentum'])
                    else:
                        p.copy_(z)

            group['k'] += 1

        return loss


class DAdaptAdam(BaseOptimizer):
    r"""Adam with D-Adaptation. Leave LR set to 1 unless you encounter instability. This implementation is based on V3.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. betas.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use AdamW style weight decay.
    :param fixed_decay: bool. fix weight decay.
    :param bias_correction: bool. Turn on Adam's bias correction.
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
        fixed_decay: bool = False,
        bias_correction: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'step': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdam'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

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
        device = group['params'][0].device

        beta1, beta2 = group['betas']

        beta2_sq: float = math.sqrt(beta2)

        d: float = group['d']
        lr: float = group['lr']

        bias_correction1: float = 1.0 - beta1 ** (group['step'] + 1)
        bias_correction2_sq: float = math.sqrt(1.0 - beta2 ** (group['step'] + 1))
        bias_correction: float = bias_correction1 / bias_correction2_sq

        # it's not Adam Debias
        d_lr: float = self.apply_adam_debias(
            not group['bias_correction'], step_size=d * lr, bias_correction1=bias_correction
        )

        sk_l1 = torch.tensor([0.0], device=device)
        numerator_acc = torch.tensor([0.0], device=device)

        if 'numerator_weighted' not in group:
            group['numerator_weighted'] = torch.tensor([0.0], device=device)
        numerator_weighted = group['numerator_weighted']

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if 'step' not in state:
                    state['s'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, s = state['exp_avg'], state['exp_avg_sq'], state['s']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                numerator_acc.add_(torch.dot(grad.flatten(), s.div(de_nom).flatten()), alpha=d_lr)

                exp_avg.mul_(beta1).add_(grad, alpha=d_lr * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                s.mul_(beta2_sq).add_(grad, alpha=d_lr * (1.0 - beta2_sq))

                sk_l1.add_(s.abs().sum())

        if sk_l1 == 0:
            return loss

        numerator_weighted.mul_(beta2_sq).add_(numerator_acc, alpha=1.0 - beta2_sq)  # fmt: skip

        if lr > 0.0:
            d_hat = numerator_weighted / (1.0 - beta2_sq) * sk_l1
            d = max(d, min(d_hat.item(), d * group['growth_rate']))

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.addcdiv_(exp_avg, de_nom, value=-1.0)

        return loss


class DAdaptSGD(BaseOptimizer):
    r"""SGD with D-Adaptation. Leave LR set to 1 unless you encounter instability. This implementation is based on V3.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        momentum: float = 0.9,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptSGD'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['z'] = p.clone()
                state['s'] = torch.zeros_like(p)
                state['x0'] = p.clone()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = group['params'][0].device

        sk_sq = torch.tensor([0.0], device=device)
        if 'numerator_weighted' not in group:
            group['numerator_weighted'] = torch.tensor([0.0], device=device)
        numerator_weighted = group['numerator_weighted']

        if group['step'] == 0:
            group['g0_norm'] = get_global_gradient_norm(self.param_groups).sqrt_().item()
        g0_norm = group['g0_norm']

        if g0_norm == 0:
            return loss

        d, lr = group['d'], group['lr']
        d_lr: float = d * lr / g0_norm

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['z'] = p.clone()
                    state['s'] = torch.zeros_like(p)
                    state['x0'] = p.clone()

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                s = state['s']
                numerator_weighted.add_(torch.dot(grad.flatten(), s.flatten()), alpha=d_lr)

                s.add_(grad, alpha=d_lr)
                sk_sq.add_(s.pow(2).sum())

        if lr > 0.0:
            d_hat = 2.0 * numerator_weighted / sk_sq.sqrt()
            d = max(d, min(d_hat.item(), d * group['growth_rate']))

        for group in self.param_groups:
            group['step'] += 1

            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                z = state['z']
                z.copy_(state['x0'] - state['s'])

                p.mul_(group['momentum']).add_(z, alpha=1.0 - group['momentum'])

        return loss


class DAdaptAdan(BaseOptimizer):
    r"""Adan with D-Adaptation. Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
        Default is inf, for unrestricted.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        betas: BETAS = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'd': d0,
            'growth_rate': growth_rate,
            'k': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdan'

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
                state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        beta1, beta2, beta3 = group['betas']
        growth_rate = group['growth_rate']

        d, lr = group['d'], group['lr']
        d_lr = float(d * lr)

        g_sq = torch.tensor([0.0], device=group['params'][0].device)
        sk_sq_weighted = torch.tensor([0.0], device=group['params'][0].device)
        sk_l1 = torch.tensor([0.0], device=group['params'][0].device)
        if 'gsq_weighted' not in group:
            group['gsq_weighted'] = torch.tensor([0.0], device=group['params'][0].device)
        gsq_weighted = group['gsq_weighted']

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
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = -grad.clone()

                grad_diff = state['previous_grad']
                grad_diff.add_(grad)

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']

                exp_avg.mul_(beta1).add_(grad, alpha=d_lr * (1.0 - beta1))
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=d_lr * (1.0 - beta2))

                grad_diff.mul_(beta2).add_(grad)
                grad_diff = to_real(grad_diff * grad_diff.conj())
                exp_avg_sq.mul_(beta3).addcmul_(grad_diff, grad_diff, value=1.0 - beta3)

                grad_power = to_real(grad * grad.conj())
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                g_sq.add_(grad_power.div_(de_nom).sum())

                s = state['s']
                s.mul_(beta3).add_(grad, alpha=d_lr * (1.0 - beta3))

                sk_sq_weighted.add_(to_real(s * s.conj()).div_(de_nom).sum())
                sk_l1.add_(s.abs().sum())

                state['previous_grad'].copy_(-grad)

        if sk_l1 == 0:
            return loss

        gsq_weighted.mul_(beta3).add_(g_sq, alpha=(d_lr ** 2) * (1.0 - beta3))  # fmt: skip

        if lr > 0.0:
            d_hat = (sk_sq_weighted / (1.0 - beta3) - gsq_weighted) / sk_l1
            d = max(d, min(d_hat, d * growth_rate))

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['step'] += 1

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                if group['weight_decouple']:
                    p.mul_(1.0 - d_lr * group['weight_decay'])

                p.addcdiv_(exp_avg, de_nom, value=-1.0)
                p.addcdiv_(exp_avg_diff, de_nom, value=-beta2)

                if not group['weight_decouple']:
                    p.div_(1.0 + d_lr * group['weight_decay'])

            group['k'] += 1

        return loss


class DAdaptLion(BaseOptimizer):
    r"""Lion with D-Adaptation. Leave LR set to 1 unless you encounter instability. This implementation is based on V3.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        betas: BETAS = (0.9, 0.999),
        d0: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptLion'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['s'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = group['params'][0].device

        if 'numerator_weighted' not in group:
            group['numerator_weighted'] = torch.tensor([0.0], device=device)
        numerator_weighted = group['numerator_weighted']

        sk_l1 = torch.tensor([0.0], device=device)
        numerator_accumulator = torch.tensor([0.0], device=device)

        beta1, beta2 = group['betas']
        beta2_sq = math.sqrt(beta2)

        d, lr = group['d'], group['lr']
        d_lr: float = d * lr

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, s = state['exp_avg'], state['s']

                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-d_lr)

                exp_avg.mul_(beta2).add_(grad, alpha=(1.0 - beta2) * d_lr)

                numerator_accumulator.add_(torch.dot(update.flatten(), s.flatten()), alpha=d_lr)
                s.mul_(beta2_sq).add_(update, alpha=(1.0 - beta2_sq) * d_lr)

                sk_l1.add_(s.abs().sum())

        numerator_weighted.mul_(beta2_sq).add_(numerator_accumulator, alpha=1.0 - beta2_sq)

        if sk_l1 == 0:
            return loss

        if lr > 0.0:
            d_hat: float = (numerator_weighted / ((1.0 - beta2_sq) * sk_l1)).item()
            d = max(d, d_hat)

        for group in self.param_groups:
            group['step'] += 1

            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

        return loss
