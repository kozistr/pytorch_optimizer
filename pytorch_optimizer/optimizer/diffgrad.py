import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import group_tensors_by_device_and_dtype


class DiffGrad(BaseOptimizer):
    """Adam updates scaled by changes between consecutive gradients.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        rectify: Perform the rectified update similar to RAdam.
        n_sma_threshold: Minimum effective simple moving average length for rectification.
        degenerated_to_sgd: Use an SGD update before the moving average reaches the rectification threshold.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
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
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'diffGrad'

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
                state['previous_grad'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return not group.get('adanorm') and self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        step_size: float,
        n_sma: float,
    ) -> None:
        beta1, beta2 = group['betas']
        exp_avgs, exp_avg_sqs = state_dict['exp_avg'], state_dict['exp_avg_sq']
        previous_grads = state_dict['previous_grad']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        if not group['rectify'] or n_sma >= self.n_sma_threshold:
            de_noms = self.apply_ams_bound_foreach(
                group['ams_bound'], exp_avg_sqs, state_dict.get('max_exp_avg_sq', []), group['eps']
            )

            torch._foreach_sub_(previous_grads, grads)
            torch._foreach_abs_(previous_grads)
            torch._foreach_sigmoid_(previous_grads)
            torch._foreach_mul_(previous_grads, exp_avgs)

            torch._foreach_addcdiv_(params, previous_grads, de_noms, value=-step_size)
        else:
            if group['ams_bound']:
                torch._foreach_maximum_(state_dict['max_exp_avg_sq'], exp_avg_sqs)

            if step_size > 0:
                torch._foreach_add_(params, exp_avgs, alpha=-step_size)

        torch._foreach_copy_(previous_grads, grads)

    def _step_per_param(self, group: ParamGroup, step_size: float, n_sma: float) -> None:
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            state = self.state[p]

            exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

            p, grad, exp_avg, exp_avg_sq, previous_grad = self.view_as_real(
                p, grad, exp_avg, exp_avg_sq, previous_grad
            )

            s_grad = self.get_adanorm_gradient(
                grad=grad,
                adanorm=group.get('adanorm', False),
                exp_grad_norm=state.get('exp_grad_adanorm', None),
                r=group.get('adanorm_r', None),
            )

            exp_avg.lerp_(s_grad, weight=1.0 - beta1)

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            de_nom = self.apply_ams_bound(
                ams_bound=group['ams_bound'],
                exp_avg_sq=exp_avg_sq,
                max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                eps=group['eps'],
            )

            dfc = previous_grad.clone()
            dfc.sub_(grad).abs_().sigmoid_().mul_(exp_avg)

            state['previous_grad'].copy_(
                torch.view_as_complex(grad) if torch.is_complex(state['previous_grad']) else grad
            )

            if not group['rectify']:
                p.addcdiv_(dfc, de_nom, value=-step_size)
                continue

            if n_sma >= self.n_sma_threshold:
                p.addcdiv_(dfc, de_nom, value=-step_size)
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

            if not group['rectify']:
                step_size *= math.sqrt(self.debias(beta2, group['step']))

            step_size = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            if self._can_use_foreach(group):
                state_keys = ['exp_avg', 'exp_avg_sq', 'previous_grad']

                if group['ams_bound']:
                    state_keys.append('max_exp_avg_sq')

                params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=state_keys)

                for tensors in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(group, tensors['params'], tensors['grads'], tensors, step_size, n_sma)
            else:
                self._step_per_param(group, step_size, n_sma)

        return loss
