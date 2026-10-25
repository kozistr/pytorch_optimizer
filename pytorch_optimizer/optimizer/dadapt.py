# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    group_tensors_by_device_and_dtype,
)
from pytorch_optimizer.optimizer.utils.gradient import get_global_gradient_norm
from pytorch_optimizer.optimizer.utils.precision import to_real


class DAdaptAdaGrad(BaseOptimizer):
    """AdaGrad with D-Adaptation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Multiplier for the adapted step size. Use `1.0` unless training is unstable.
        momentum: Momentum factor.
        d0: Initial estimate of the distance to the optimum.
        growth_rate: Maximum multiplicative growth of the distance estimate per step.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported CUDA parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 0.0,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self._compiled_update_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'k': 0,
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdaGrad'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if 'alpha_k' not in state:
                state['alpha_k'] = torch.full_like(p, fill_value=1e-6)
                state['sk'] = torch.zeros_like(p)
                state['x0'] = torch.clone(p)
                if p.grad.is_sparse:
                    state['weighted_sk'] = torch.zeros_like(p)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return self.can_use_foreach(group, group.get('foreach')) and (
            group.get('foreach') is not None or any(p.is_cuda for p in group['params'] if p.grad is not None)
        )

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        super()._compile_foreach(compile_kwargs)
        self._compiled_update_foreach = compile_foreach_step(self._apply_update_foreach, compile_kwargs)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        g_sq, sk_sq_weighted_change, sk_l1_change = statistics
        sk, alpha_k = state['sk'], state['alpha_k']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        de_noms = torch._foreach_sqrt(alpha_k)
        torch._foreach_add_(de_noms, group['eps'])
        old_sk_sq_weighted = torch._foreach_div(torch._foreach_pow(sk, 2), de_noms)
        old_sk_abs = torch._foreach_abs(sk)

        grad_sq = torch._foreach_pow(grads, 2)
        torch._foreach_add_(alpha_k, grad_sq)
        de_noms = torch._foreach_sqrt(alpha_k)
        torch._foreach_add_(de_noms, group['eps'])
        torch._foreach_div_(grad_sq, de_noms)

        foreach_add_(sk, grads, alpha=d_lr)
        sk_sq_weighted = torch._foreach_div(torch._foreach_pow(sk, 2), de_noms)
        sk_abs = torch._foreach_abs(sk)

        for grad_power, old_weighted, weighted, old_abs, new_abs in zip(
            grad_sq, old_sk_sq_weighted, sk_sq_weighted, old_sk_abs, sk_abs
        ):
            g_sq.add_(grad_power.sum())
            sk_sq_weighted_change.add_(weighted.sum() - old_weighted.sum())
            sk_l1_change.add_(new_abs.sum() - old_abs.sum())

    def _apply_update_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        de_noms = torch._foreach_sqrt(state['alpha_k'])
        torch._foreach_add_(de_noms, group['eps'])
        updates = torch._foreach_sub(state['x0'], torch._foreach_div(state['sk'], de_noms))

        if group['momentum'] > 0.0:
            torch._foreach_lerp_(params, updates, weight=1.0 - group['momentum'])
        else:
            torch._foreach_copy_(params, updates)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:  # noqa: PLR0912, PLR0915
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = self.get_device_from_param_groups(self.param_groups)

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
            self.init_group(group)
            group['step'] += 1

            eps = group['eps']

            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(group, self.state, ['sk', 'alpha_k'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch, d_lr,
                        (g_sq, sk_sq_weighted_change, sk_l1_change),
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

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
                    alpha_k_delta = torch.sparse_coo_tensor(
                        grad.indices(),
                        alpha_k_delta_masked,
                        grad.shape,
                        check_invariants=False,
                    )
                    alpha_k.add_(alpha_k_delta)

                    de_nom = torch.sqrt(alpha_k_p1_masked + eps)

                    grad_sq = vk.div(de_nom).sum()
                    g_sq.add_(grad_sq)

                    # update weighted sk sq tracking
                    weighted_sk_p1_masked = sk_masked._values().pow(2).div(de_nom)

                    sk_sq_weighted_change.add_(weighted_sk_p1_masked.sum() - weighted_sk_masked._values().sum())

                    weighted_sk_p1_delta_masked = weighted_sk_p1_masked - weighted_sk_masked._values()
                    weighted_sk_p1_delta = torch.sparse_coo_tensor(
                        grad.indices(),
                        weighted_sk_p1_delta_masked,
                        grad.shape,
                        check_invariants=False,
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

            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(group, self.state, ['sk', 'alpha_k', 'x0'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._compiled_update_foreach or self._apply_update_foreach
                    update(group, batch['params'], batch['grads'], batch)

                group['k'] += 1
                continue

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
                    loc_delta = torch.sparse_coo_tensor(
                        grad.indices(),
                        loc_delta_masked,
                        grad.shape,
                        check_invariants=False,
                    )
                    p.add_(loc_delta)
                else:
                    z = x0 - sk.div(alpha_k.sqrt().add_(group['eps']))

                    if group['momentum'] > 0.0:
                        p.lerp_(z, weight=1.0 - group['momentum'])
                    else:
                        p.copy_(z)

            group['k'] += 1

        return loss


class DAdaptAdam(BaseOptimizer):
    """Adam with D-Adaptation V3.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Multiplier for the adapted step size. Use `1.0` unless training is unstable.
        betas: Decay rates for the gradient mean and squared gradients.
        d0: Initial estimate of the distance to the optimum.
        growth_rate: Maximum multiplicative growth of the distance estimate per step.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        bias_correction: Apply bias correction to the moment estimates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1.0,
        betas: Betas = (0.9, 0.999),
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self._compiled_update_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'step': 0,
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdam'

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
                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        super()._compile_foreach(compile_kwargs)
        self._compiled_update_foreach = compile_foreach_step(self._apply_update_foreach, compile_kwargs)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        numerator_acc, sk_l1 = statistics
        beta1, beta2 = group['betas']

        if self.maximize:
            torch._foreach_neg_(grads)

        if not group['weight_decouple']:
            self.apply_weight_decay_foreach(params, grads, d_lr, group['weight_decay'], False, group['fixed_decay'])

        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])

        for grad, s, de_nom in zip(grads, state['s'], de_noms):
            value = torch.dot(grad.flatten(), s.div(de_nom).flatten())
            foreach_add_([numerator_acc], [value], alpha=d_lr)

        torch._foreach_mul_(state['exp_avg'], beta1)
        foreach_add_(state['exp_avg'], grads, alpha=d_lr * (1.0 - beta1))
        torch._foreach_mul_(state['exp_avg_sq'], beta2)
        torch._foreach_addcmul_(state['exp_avg_sq'], grads, grads, value=1.0 - beta2)

        torch._foreach_mul_(state['s'], math.sqrt(beta2))
        foreach_add_(state['s'], grads, alpha=d_lr * (1.0 - math.sqrt(beta2)))

        for s in state['s']:
            sk_l1.add_(s.abs().sum())

    def _apply_update_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
    ) -> None:
        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])

        if group['weight_decouple']:
            self.apply_weight_decay_foreach(params, grads, d_lr, group['weight_decay'], True, group['fixed_decay'])

        torch._foreach_addcdiv_(params, state['exp_avg'], de_noms, value=-1.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = self.get_device_from_param_groups(self.param_groups)

        beta1, beta2 = group['betas']

        beta2_sq: float = math.sqrt(beta2)

        d: float = group['d']
        lr: float = group['lr']

        bias_correction1: float = 1.0 - beta1 ** (group['step'] + 1)
        bias_correction2_sq: float = math.sqrt(1.0 - beta2 ** (group['step'] + 1))
        bias_correction: float = bias_correction1 / bias_correction2_sq

        d_lr: float = self.apply_adam_debias(
            not group['bias_correction'], step_size=d * lr, bias_correction1=bias_correction
        )

        sk_l1 = torch.tensor([0.0], device=device)
        numerator_acc = torch.tensor([0.0], device=device)

        if 'numerator_weighted' not in group:
            group['numerator_weighted'] = torch.tensor([0.0], device=device)
        numerator_weighted = group['numerator_weighted']

        for group in self.param_groups:
            self.init_group(group)

            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['exp_avg', 'exp_avg_sq', 's'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        {**group, 'betas': (beta1, beta2)},
                        batch['params'],
                        batch['grads'],
                        batch,
                        d_lr,
                        (numerator_acc, sk_l1),
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, s = state['exp_avg'], state['exp_avg_sq'], state['s']

                if not group['weight_decouple']:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=d_lr,
                        weight_decay=group['weight_decay'],
                        weight_decouple=False,
                        fixed_decay=group['fixed_decay'],
                    )

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                numerator_acc.add_(torch.dot(grad.flatten(), s.div(de_nom).flatten()), alpha=d_lr)

                exp_avg.mul_(beta1).add_(grad, alpha=d_lr * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                s.mul_(beta2_sq).add_(grad, alpha=d_lr * (1.0 - beta2_sq))

                sk_l1.add_(s.abs().sum())

        if sk_l1 == 0:
            return loss

        numerator_weighted.lerp_(numerator_acc, weight=1.0 - beta2_sq)  # fmt: skip

        if lr > 0.0:
            d_hat = numerator_weighted / ((1.0 - beta2_sq) * sk_l1)
            d = max(d, min(d_hat.item(), d * group['growth_rate']))

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['exp_avg', 'exp_avg_sq'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._compiled_update_foreach or self._apply_update_foreach
                    update(group, batch['params'], batch['grads'], batch, d_lr)

                continue

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
    """SGD with D-Adaptation V3.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Multiplier for the adapted step size. Use `1.0` unless training is unstable.
        momentum: Momentum factor.
        d0: Initial estimate of the distance to the optimum.
        growth_rate: Maximum multiplicative growth of the distance estimate per step.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1.0,
        momentum: float = 0.9,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self._compiled_update_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptSGD'

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
                state['z'] = p.clone()
                state['s'] = torch.zeros_like(p)
                state['x0'] = p.clone()

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        super()._compile_foreach(compile_kwargs)
        self._compiled_update_foreach = compile_foreach_step(self._apply_update_foreach, compile_kwargs)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        numerator_weighted, sk_sq = statistics

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, d_lr, group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        for grad, s in zip(grads, state['s']):
            value = torch.dot(grad.flatten(), s.flatten())
            foreach_add_([numerator_weighted], [value], alpha=d_lr)

        foreach_add_(state['s'], grads, alpha=d_lr)

        for s in state['s']:
            sk_sq.add_(s.pow(2).sum())

    def _apply_update_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
    ) -> None:
        torch._foreach_copy_(state['z'], torch._foreach_sub(state['x0'], state['s']))
        torch._foreach_lerp_(params, state['z'], weight=1.0 - group['momentum'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = self.get_device_from_param_groups(self.param_groups)

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
            self.init_group(group)

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['s'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch,
                        d_lr,
                        (numerator_weighted, sk_sq),
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

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['z', 'x0', 's'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._compiled_update_foreach or self._apply_update_foreach
                    update(group, batch['params'], batch['grads'], batch)

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                z = state['z']
                z.copy_(state['x0'] - state['s'])

                p.lerp_(z, weight=1.0 - group['momentum'])

        return loss


class DAdaptAdan(BaseOptimizer):
    """Adan with D-Adaptation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Multiplier for the adapted step size. Use `1.0` unless training is unstable.
        betas: Decay rates for gradients, gradient differences, and squared corrected gradients.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        d0: Initial estimate of the distance to the optimum.
        growth_rate: Maximum multiplicative growth of the distance estimate per step.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported CUDA parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self._compiled_update_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'd': d0,
            'growth_rate': growth_rate,
            'k': 0,
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdan'

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

            if 'exp_avg' not in state:
                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = -grad.clone()

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return self.can_use_foreach(group, group.get('foreach')) and (
            group.get('foreach') is not None or any(p.is_cuda for p in group['params'] if p.grad is not None)
        ) and all(
            p.dtype not in (torch.float16, torch.bfloat16) for p in group['params'] if p.grad is not None
        )

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        super()._compile_foreach(compile_kwargs)
        self._compiled_update_foreach = compile_foreach_step(self._apply_update_foreach, compile_kwargs)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        g_sq, sk_sq_weighted, sk_l1 = statistics
        beta1, beta2, beta3 = group['betas']
        grad_diff = state['previous_grad']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_add_(grad_diff, grads)
        torch._foreach_mul_(state['exp_avg'], beta1)
        foreach_add_(state['exp_avg'], grads, alpha=d_lr * (1.0 - beta1))
        torch._foreach_mul_(state['exp_avg_diff'], beta2)
        foreach_add_(state['exp_avg_diff'], grad_diff, alpha=d_lr * (1.0 - beta2))

        torch._foreach_mul_(grad_diff, beta2)
        torch._foreach_add_(grad_diff, grads)
        grad_power = torch._foreach_mul(grad_diff, grad_diff)
        torch._foreach_mul_(state['exp_avg_sq'], beta3)
        torch._foreach_addcmul_(state['exp_avg_sq'], grad_power, grad_power, value=1.0 - beta3)

        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])
        grad_power = torch._foreach_div(torch._foreach_mul(grads, grads), de_noms)

        torch._foreach_mul_(state['s'], beta3)
        foreach_add_(state['s'], grads, alpha=d_lr * (1.0 - beta3))
        weighted_sk = torch._foreach_div(torch._foreach_mul(state['s'], state['s']), de_noms)
        sk_abs = torch._foreach_abs(state['s'])

        for power, weighted, absolute in zip(grad_power, weighted_sk, sk_abs):
            g_sq.add_(power.sum())
            sk_sq_weighted.add_(weighted.sum())
            sk_l1.add_(absolute.sum())

        torch._foreach_copy_(state['previous_grad'], torch._foreach_neg(grads))

    def _apply_update_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
    ) -> None:
        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])

        if group['weight_decouple']:
            torch._foreach_mul_(params, 1.0 - d_lr * group['weight_decay'])

        torch._foreach_addcdiv_(params, state['exp_avg'], de_noms, value=-1.0)
        torch._foreach_addcdiv_(params, state['exp_avg_diff'], de_noms, value=-group['betas'][1])

        if not group['weight_decouple']:
            torch._foreach_div_(params, 1.0 + d_lr * group['weight_decay'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:  # noqa: PLR0915
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        beta1, beta2, beta3 = group['betas']
        growth_rate = group['growth_rate']

        d, lr = group['d'], group['lr']
        d_lr = float(d * lr)

        device = self.get_device_from_param_groups(self.param_groups)
        g_sq = torch.tensor([0.0], device=device)
        sk_sq_weighted = torch.tensor([0.0], device=device)
        sk_l1 = torch.tensor([0.0], device=device)

        if 'gsq_weighted' not in group:
            group['gsq_weighted'] = torch.tensor([0.0], device=device)

        gsq_weighted = group['gsq_weighted']

        for group in self.param_groups:
            self.init_group(group)

            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, ['exp_avg', 'exp_avg_sq', 'exp_avg_diff', 's', 'previous_grad']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        {**group, 'betas': (beta1, beta2, beta3)},
                        batch['params'], batch['grads'], batch, d_lr, (g_sq, sk_sq_weighted, sk_l1),
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

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
            group['step'] += 1

            group['gsq_weighted'] = gsq_weighted
            group['d'] = d

            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, ['exp_avg', 'exp_avg_sq', 'exp_avg_diff']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._compiled_update_foreach or self._apply_update_foreach
                    update({**group, 'betas': (beta1, beta2, beta3)}, batch['params'], batch['grads'], batch, d_lr)

                group['k'] += 1
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

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
    """Lion with D-Adaptation V3.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Multiplier for the adapted step size. Use `1.0` unless training is unstable.
        betas: Decay rates for update interpolation and gradient momentum.
        d0: Initial estimate of the distance to the optimum.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1.0,
        betas: Betas = (0.9, 0.999),
        d0: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptLion'

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
                state['s'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        d_lr: float | torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        numerator_accumulator, sk_l1 = statistics
        beta1, beta2 = group['betas']
        beta2_sq = math.sqrt(beta2)

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, d_lr, group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        updates = torch._foreach_lerp(state['exp_avg'], grads, weight=1.0 - beta1)
        torch._foreach_sign_(updates)
        foreach_add_(params, updates, alpha=-d_lr)

        torch._foreach_mul_(state['exp_avg'], beta2)
        foreach_add_(state['exp_avg'], grads, alpha=(1.0 - beta2) * d_lr)

        for update, s in zip(updates, state['s']):
            value = torch.dot(update.flatten(), s.flatten())
            foreach_add_([numerator_accumulator], [value], alpha=d_lr)

        torch._foreach_mul_(state['s'], beta2_sq)
        foreach_add_(state['s'], updates, alpha=(1.0 - beta2_sq) * d_lr)

        for s in state['s']:
            sk_l1.add_(s.abs().sum())

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = self.get_device_from_param_groups(self.param_groups)

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
            self.init_group(group)

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['exp_avg', 's'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        {**group, 'betas': (beta1, beta2)},
                        batch['params'],
                        batch['grads'],
                        batch,
                        d_lr,
                        (numerator_accumulator, sk_l1),
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
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, s = state['exp_avg'], state['s']

                update = exp_avg.lerp(grad, weight=1.0 - beta1).sign_()
                p.add_(update, alpha=-d_lr)

                exp_avg.mul_(beta2).add_(grad, alpha=(1.0 - beta2) * d_lr)

                numerator_accumulator.add_(torch.dot(update.flatten(), s.flatten()), alpha=d_lr)
                s.mul_(beta2_sq).add_(update, alpha=(1.0 - beta2_sq) * d_lr)

                sk_l1.add_(s.abs().sum())

        numerator_weighted.lerp_(numerator_accumulator, weight=1.0 - beta2_sq)

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
