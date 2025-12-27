import math
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.foreach_utils import (
    foreach_add_,
    foreach_addcdiv_,
    foreach_addcmul_,
    foreach_lerp_,
    foreach_mul_,
    foreach_sqrt,
)


class AdaBelief(BaseOptimizer):
    """Adapting Step-sizes by the Belief in Observed Gradients.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        rectify (bool): Perform the rectified update similar to RAdam.
        n_sma_threshold: Number of SMA threshold (recommended is 5).
        degenerated_to_sgd (bool): Perform SGD update when variance of gradient is high.
        ams_bound (bool): Whether to use the AMSBound variant.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        ams_bound: bool = False,
        foreach: Optional[bool] = None,
        eps: float = 1e-16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'ams_bound': ams_bound,
            'foreach': foreach,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaBelief'

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
                state['exp_avg_var'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        if group.get('adanorm') or group['rectify'] or group['ams_bound']:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_vars: List[torch.Tensor],
    ) -> None:
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']

        bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
            foreach=True,
        )

        foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1, foreach=True)

        grad_residuals = [g.sub(ea) for g, ea in zip(grads, exp_avgs)]

        foreach_mul_(exp_avg_vars, beta2, foreach=True)
        foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1.0 - beta2, foreach=True)
        foreach_add_(exp_avg_vars, eps, foreach=True)

        de_noms = foreach_sqrt(exp_avg_vars, foreach=True)
        for d in de_noms:
            d.div_(bias_correction2_sq)

        foreach_addcdiv_(params, exp_avgs, de_noms, value=-lr, foreach=True)

    def _step_per_param(self, group: ParamGroup, step_size: float, n_sma: float) -> None:
        beta1, beta2 = group['betas']

        bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

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

            exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

            p, grad, exp_avg, exp_avg_var = self.view_as_real(p, grad, exp_avg, exp_avg_var)

            s_grad = self.get_adanorm_gradient(
                grad=grad,
                adanorm=group.get('adanorm', False),
                exp_grad_norm=state.get('exp_grad_adanorm', None),
                r=group.get('adanorm_r', None),
            )

            exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)

            grad_residual = grad - exp_avg
            exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2).add_(group['eps'])

            de_nom = self.apply_ams_bound(
                ams_bound=group['ams_bound'],
                exp_avg_sq=exp_avg_var,
                max_exp_avg_sq=state.get('max_exp_avg_var', None),
                eps=group['eps'],
            )

            if not group['rectify']:
                de_nom.div_(bias_correction2_sq)
                p.addcdiv_(exp_avg, de_nom, value=-step_size)
                continue

            if n_sma >= self.n_sma_threshold:
                p.addcdiv_(exp_avg, de_nom, value=-step_size)
            elif step_size > 0:
                p.add_(exp_avg, alpha=-step_size)

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

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group['rectify'],
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_var']
                )
                if params:
                    self._step_foreach(group, params, grads, state_dict['exp_avg'], state_dict['exp_avg_var'])
            else:
                self._step_per_param(group, step_size, n_sma)

        return loss
