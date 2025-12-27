import math
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt_


class Amos(BaseOptimizer):
    """An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        beta (float): A float slightly less than 1. Recommended to set `1 - beta` approximately the same magnitude
            as the learning rate, similar to beta2 in Adam.
        momentum (float): Exponential decay rate for optional moving average of updates.
        extra_l2 (float): Additional L2 regularization.
        c_coef (float): Coefficient for decay_factor_c.
        d_coef (float): Coefficient for decay_factor_d.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        foreach: Optional[bool] = None,
        eps: float = 1e-18,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(extra_l2, 'extra_l2')
        self.validate_non_negative(eps, 'eps')

        self.c_coef = c_coef
        self.d_coef = d_coef
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'momentum': momentum,
            'extra_l2': extra_l2,
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Amos'

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
                state['exp_avg_sq'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                state['decay'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['momentum'] > 0.0:
                    state['exp_avg'] = torch.zeros_like(p)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        r"""Get expected scale for model weights."""
        if len(p.shape) == 1:
            return 0.5
        if len(p.shape) == 2:
            return math.sqrt(2 / p.size(1))
        return math.sqrt(1 / p.size(1))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        decays: List[torch.Tensor],
    ) -> None:
        lr_sq: float = math.sqrt(group['lr'])
        lr_p2: float = math.pow(group['lr'], 2)

        beta: float = group['beta']
        bias_correction: float = self.debias(beta, group['step'])

        if self.maximize:
            torch._foreach_neg_(grads)

        g2 = [grad.pow(2).mean() for grad in grads]
        init_lrs: List[float] = [group['lr'] * self.get_scale(p) for p in params]

        torch._foreach_mul_(exp_avg_sqs, beta)
        torch._foreach_add_(exp_avg_sqs, g2, alpha=1.0 - beta)

        r_v_hat = torch._foreach_add(exp_avg_sqs, group['eps'])
        torch._foreach_reciprocal_(r_v_hat)
        torch._foreach_mul_(r_v_hat, bias_correction)

        df_c = torch._foreach_mul(decays, self.c_coef * lr_sq)
        torch._foreach_add_(df_c, 1.0)
        foreach_rsqrt_(df_c)

        d_step_sizes = [self.d_coef * math.sqrt(step_size) for step_size in init_lrs]
        df_d = torch._foreach_mul(decays, d_step_sizes)
        torch._foreach_add_(df_d, 1.0)

        torch._foreach_mul_(df_c, r_v_hat)
        torch._foreach_mul_(df_c, lr_p2)
        torch._foreach_mul_(df_c, g2)

        updates = torch._foreach_div(params, 2.0)
        torch._foreach_mul_(updates, torch._foreach_sub(df_c, group['extra_l2']))

        torch._foreach_sqrt_(r_v_hat)
        torch._foreach_mul_(r_v_hat, init_lrs)
        torch._foreach_add_(updates, torch._foreach_mul(grads, r_v_hat))

        torch._foreach_div_(updates, df_d)

        torch._foreach_mul_(decays, torch._foreach_add(df_c, 1.0))
        torch._foreach_add_(decays, df_c)

        if group['momentum'] > 0.0:
            torch._foreach_lerp_(exp_avgs, updates, weight=1.0 - group['momentum'])
            torch._foreach_copy_(updates, exp_avgs)

        torch._foreach_sub_(params, updates)

    def _step_per_param(self, group: ParamGroup) -> None:
        momentum, beta = group['momentum'], group['beta']

        lr_sq: float = math.sqrt(group['lr'])
        lr_p2: float = math.pow(group['lr'], 2)
        bias_correction: float = self.debias(beta, group['step'])

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            g2 = grad.pow(2).mean()
            init_lr: float = group['lr'] * self.get_scale(p)

            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul_(beta).add_(g2, alpha=1.0 - beta)

            r_v_hat = bias_correction / (exp_avg_sq + group['eps'])

            decay = state['decay']
            decay_factor_c = torch.rsqrt(1.0 + self.c_coef * lr_sq * decay)
            decay_factor_d = torch.reciprocal(1.0 + self.d_coef * math.sqrt(init_lr) * decay)

            gamma = decay_factor_c * lr_p2 * r_v_hat * g2

            update = p.clone()
            update.mul_((gamma - group['extra_l2']) / 2.0)
            update.add_(r_v_hat.sqrt() * grad, alpha=init_lr)
            update.mul_(decay_factor_d)

            decay.mul_(1.0 + gamma).add_(gamma)

            if momentum > 0.0:
                exp_avg = state['exp_avg']
                exp_avg.mul_(momentum).add_(update, alpha=1.0 - momentum)

                update.copy_(exp_avg)

            p.add_(-update)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq', 'decay']
                )
                if params:
                    self._step_foreach(
                        group,
                        params,
                        grads,
                        state_dict['exp_avg'],
                        state_dict['exp_avg_sq'],
                        state_dict['decay'],
                    )
            else:
                self._step_per_param(group)

        return loss
