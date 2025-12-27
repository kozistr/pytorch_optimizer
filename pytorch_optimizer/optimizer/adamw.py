import math
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class StableAdamW(BaseOptimizer):
    """Stable and low-precision training for large-scale vision-language models.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        kahan_sum (bool): Enables Kahan summation for more accurate parameter updates when training in low precision
            (float16 or bfloat16).
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Decoupled weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.99),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'kahan_sum': kahan_sum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableAdamW'

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

                state['kahan_comp'] = (
                    torch.zeros_like(p)
                    if (group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16})
                    else None
                )

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        """Check if foreach can be used for this group.

        Foreach is disabled when using features that require per-parameter handling:
        - Complex tensors (view_as_real)
        """
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        kahan_comps: List[torch.Tensor],
    ) -> None:
        """Foreach-optimized step for a parameter group."""
        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']

        beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step'])
        beta2_hat: float = self.debias_beta(beta2, group['step'])

        eps_p2: float = math.pow(eps, 2)

        if self.maximize:
            torch._foreach_neg_(grads)

        step_sizes: List[float] = [
            lr / self.get_stable_adamw_rms(grad, exp_avg_sq, eps=eps_p2)
            for grad, exp_avg_sq in zip(grads, exp_avg_sqs)
        ]

        if group['weight_decay'] != 0.0:
            for p, step_size in zip(params, step_sizes):
                if group['weight_decouple']:
                    p.mul_(1.0 - group['weight_decay'] * step_size)

        torch._foreach_lerp_(exp_avgs, grads, weight=beta1_comp)

        torch._foreach_mul_(exp_avg_sqs, beta2_hat)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2_hat)

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(de_noms, eps)

        step_sizes = [-step_size for step_size in step_sizes]

        if group['kahan_sum'] and params[0].dtype in {torch.float16, torch.bfloat16}:
            de_noms = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(de_noms, group['eps'])

            torch._foreach_addcdiv_(kahan_comps, exp_avgs, de_noms, step_sizes)

            with torch.no_grad():
                torch._foreach_copy_(grads, params)

            torch._foreach_add_(params, kahan_comps)

            torch._foreach_sub_(grads, params)
            torch._foreach_add_(kahan_comps, grads)
        else:
            torch._foreach_addcdiv_(params, exp_avgs, de_noms, step_sizes)

    def _step_per_param(self, group: ParamGroup) -> None:
        """Per-parameter step (original implementation)."""
        beta1, beta2 = group['betas']

        beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step'])
        beta2_hat: float = self.debias_beta(beta2, group['step'])

        eps_p2: float = math.pow(group['eps'], 2)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            state = self.state[p]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

            exp_avg.lerp_(grad, weight=beta1_comp)
            exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1.0 - beta2_hat)

            lr: float = group['lr'] / self.get_stable_adamw_rms(grad, exp_avg_sq, eps=eps_p2)

            self.apply_weight_decay(
                p,
                grad=grad,
                lr=lr,
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

            if group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16}:
                kahan_comp = state['kahan_comp']
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-lr)

                grad.copy_(p.detach())
                p.add_(kahan_comp)

                kahan_comp.add_(grad.sub_(p))
            else:
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-lr)

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
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq', 'kahan_comp']
                )
                if params:
                    self._step_foreach(
                        group,
                        params,
                        grads,
                        state_dict['exp_avg'],
                        state_dict['exp_avg_sq'],
                        state_dict['kahan_comp'],
                    )
            else:
                self._step_per_param(group)

        return loss
