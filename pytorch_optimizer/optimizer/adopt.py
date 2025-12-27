import math
from typing import Callable, List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class ADOPT(BaseOptimizer):
    """Modified Adam Can Converge with Any β2 with the Optimal Rate.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        clip_lambda (Callable[[float], float]): Function to clip gradient. Default is `step ** 0.25`.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.9999),
        clip_lambda: Callable[[float], float] = lambda step: math.pow(step, 0.25),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        foreach: Optional[bool] = None,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.clip_lambda = clip_lambda
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPT'

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

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        if group.get('cautious') or group.get('stable_adamw'):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
    ) -> None:
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        if group['step'] == 1:
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads)
            return

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_clamp_min_(de_noms, eps)

        normed_grads = torch._foreach_div(grads, de_noms)
        if self.clip_lambda is not None:
            clip: float = self.clip_lambda(group['step'])
            torch._foreach_clamp_min_(normed_grads, -clip)
            torch._foreach_clamp_max_(normed_grads, clip)

        torch._foreach_lerp_(exp_avgs, normed_grads, weight=1.0 - beta1)

        torch._foreach_add_(params, exp_avgs, alpha=-lr)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    def _step_per_param(self, group: ParamGroup) -> None:
        beta1, beta2 = group['betas']
        lr: float = group['lr']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=lr,
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            if group['step'] == 1:
                exp_avg_sq.addcmul_(grad, grad.conj())
                continue

            de_nom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

            normed_grad = grad.div(de_nom)
            if self.clip_lambda is not None:
                clip = self.clip_lambda(group['step'])
                normed_grad.clamp_(-clip, clip)

            exp_avg.lerp_(normed_grad, weight=1.0 - beta1)

            if group.get('cautious'):
                update = exp_avg.clone()
                self.apply_cautious(update, normed_grad)
            else:
                update = exp_avg

            step_lr = lr
            if group.get('stable_adamw'):
                step_lr /= self.get_stable_adamw_rms(grad, exp_avg_sq)

            p.add_(update, alpha=-step_lr)

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)

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
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                if params:
                    self._step_foreach(group, params, grads, state_dict['exp_avg'], state_dict['exp_avg_sq'])
            else:
                self._step_per_param(group)

        return loss
