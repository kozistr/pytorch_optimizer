import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import group_tensors_by_device_and_dtype


class StableAdamW(BaseOptimizer):
    """AdamW with update clipping and optional low precision Kahan summation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        kahan_sum: Enables Kahan summation for more accurate parameter updates when training in low precision
            (float16 or bfloat16). Float16 parameters use float32 second moments to preserve squared gradients.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        eps: Term added to the denominator to improve numerical stability.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.99),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        foreach: bool | None = None,
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
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32 if p.dtype == torch.float16 else p.dtype)

                state['kahan_comp'] = (
                    torch.zeros_like(p)
                    if (group['kahan_sum'] and p.dtype in {torch.float16, torch.bfloat16})
                    else None
                )

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)

        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for p, saved_id in zip(group['params'], saved_group['params']):
                saved_state = state_dict['state'].get(saved_id, {})
                if p.dtype == torch.float16 and 'exp_avg_sq' in saved_state:
                    self.state[p]['exp_avg_sq'] = saved_state['exp_avg_sq'].to(device=p.device, dtype=torch.float32)

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
        kahan_comps: list[torch.Tensor],
    ) -> None:
        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']

        beta1_comp: float = 1.0 - self.debias_beta(beta1, group['step'])
        beta2_hat: float = self.debias_beta(beta2, group['step'])

        eps_p2: float = math.pow(eps, 2)

        if self.maximize:
            torch._foreach_neg_(grads)

        if not group['weight_decouple']:
            self.apply_weight_decay_foreach(
                params, grads, lr, group['weight_decay'], weight_decouple=False, fixed_decay=False
            )

        torch._foreach_lerp_(exp_avgs, grads, weight=beta1_comp)

        stats_grads = [grad.float() for grad in grads] if params[0].dtype == torch.float16 else grads
        torch._foreach_mul_(exp_avg_sqs, beta2_hat)
        torch._foreach_addcmul_(exp_avg_sqs, stats_grads, stats_grads, value=1.0 - beta2_hat)

        step_sizes: list[torch.Tensor] = [
            -lr / self.get_stable_adamw_rms(grad, exp_avg_sq, eps=eps_p2)
            for grad, exp_avg_sq in zip(stats_grads, exp_avg_sqs)
        ]

        if group['weight_decay'] != 0.0 and group['weight_decouple']:
            wd_step_sizes = [1.0 + group['weight_decay'] * step_size for step_size in step_sizes]
            torch._foreach_mul_(params, wd_step_sizes)

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(de_noms, eps)
        de_noms = [de_nom.to(dtype=step_size.dtype) for de_nom, step_size in zip(de_noms, step_sizes)]
        torch._foreach_div_(de_noms, step_sizes)

        if group['kahan_sum'] and params[0].dtype in (torch.float16, torch.bfloat16):
            torch._foreach_addcdiv_(kahan_comps, exp_avgs, de_noms)

            with torch.no_grad():
                torch._foreach_copy_(grads, params)

            torch._foreach_add_(params, kahan_comps)

            torch._foreach_sub_(grads, params)
            torch._foreach_add_(kahan_comps, grads)
        else:
            torch._foreach_addcdiv_(params, exp_avgs, de_noms)

    def _step_per_param(self, group: ParamGroup) -> None:
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

            self.maximize_gradient(grad, maximize=self.maximize)

            if not group['weight_decouple']:
                self.apply_weight_decay(
                    p, grad, group['lr'], group['weight_decay'], weight_decouple=False, fixed_decay=False
                )

            exp_avg.lerp_(grad, weight=beta1_comp)
            stats_grad = grad.float() if grad.dtype == torch.float16 else grad
            exp_avg_sq.mul_(beta2_hat).addcmul_(stats_grad, stats_grad, value=1.0 - beta2_hat)

            lr = group['lr'] / self.get_stable_adamw_rms(stats_grad, exp_avg_sq, eps=eps_p2)

            if group['weight_decouple']:
                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

            de_nom = exp_avg_sq.sqrt().add_(group['eps']).to(dtype=lr.dtype).div_(-lr)

            if group['kahan_sum'] and p.dtype in (torch.float16, torch.bfloat16):
                kahan_comp = state['kahan_comp']
                kahan_comp.addcdiv_(exp_avg, de_nom)

                grad.copy_(p.detach())
                p.add_(kahan_comp)

                kahan_comp.add_(grad.sub_(p))
            else:
                p.addcdiv_(exp_avg, de_nom)

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
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch['exp_avg'],
                        batch['exp_avg_sq'],
                        batch['kahan_comp'],
                    )
            else:
                self._step_per_param(group)

        return loss
