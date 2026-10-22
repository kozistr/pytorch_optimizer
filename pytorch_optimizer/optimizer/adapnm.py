import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcdiv_, group_tensors_by_device_and_dtype


class AdaPNM(BaseOptimizer):
    """Adam with positive negative momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Momentum decay, second moment decay, and positive negative momentum mixing coefficient.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = True,
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
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
            'ams_bound': ams_bound,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaPNM'

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
                state['neg_exp_avg'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2_sq: float | torch.Tensor,
        step_size: float | torch.Tensor,
    ) -> None:
        beta1, beta2, beta3 = group['betas']
        exp_avgs, neg_exp_avgs = state['exp_avg'], state['neg_exp_avg']
        exp_avg_sqs = state['exp_avg_sq']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1 ** 2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        de_noms = self.apply_ams_bound_foreach(
            group['ams_bound'], exp_avg_sqs, state.get('max_exp_avg_sq', []), group['eps']
        )
        torch._foreach_div_(de_noms, bias_correction2_sq)

        updates = torch._foreach_mul(exp_avgs, 1.0 + beta3)
        torch._foreach_add_(updates, neg_exp_avgs, alpha=-beta3)
        torch._foreach_mul_(updates, 1.0 / math.sqrt((1 + beta3) ** 2 + beta3 ** 2))
        foreach_addcdiv_(params, updates, de_noms, value=-step_size)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2, beta3 = group['betas']

            beta1_p2: float = beta1 ** 2  # fmt: skip
            noise_norm: float = math.sqrt((1 + beta3) ** 2 + beta3 ** 2)  # fmt: skip

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False), step_size=group['lr'], bias_correction1=bias_correction1
            )

            if not group.get('adanorm') and self.can_use_foreach(group, group.get('foreach', self.foreach)):
                state_keys = ['exp_avg', 'neg_exp_avg', 'exp_avg_sq']
                if group['ams_bound']:
                    state_keys.append('max_exp_avg_sq')
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=state_keys)
                if group['step'] % 2 == 0:
                    state['exp_avg'], state['neg_exp_avg'] = state['neg_exp_avg'], state['exp_avg']
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, bias_correction2_sq, step_size)
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
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg_sq = state['exp_avg_sq']

                if group['step'] % 2 == 1:
                    exp_avg, neg_exp_avg = state['exp_avg'], state['neg_exp_avg']
                else:
                    exp_avg, neg_exp_avg = state['neg_exp_avg'], state['exp_avg']

                p, grad, exp_avg, neg_exp_avg, exp_avg_sq = self.view_as_real(
                    p, grad, exp_avg, neg_exp_avg, exp_avg_sq
                )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg.lerp_(s_grad, weight=1.0 - beta1_p2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )
                de_nom.div_(bias_correction2_sq)

                pn_momentum = exp_avg.mul(1.0 + beta3).add_(neg_exp_avg, alpha=-beta3).mul_(1.0 / noise_norm)

                p.addcdiv_(pn_momentum, de_nom, value=-step_size)

        return loss
