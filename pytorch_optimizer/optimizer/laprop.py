import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcdiv_, group_tensors_by_device_and_dtype


class LaProp(BaseOptimizer):
    """Adaptive updates with momentum of preconditioned gradients.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        centered: Subtract the squared gradient mean from the second moment estimate.
        steps_before_using_centered: Number of steps to accumulate gradient means before applying centered updates.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Epsilon value for numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 4e-4,
        betas: Betas = (0.9, 0.999),
        centered: bool = False,
        steps_before_using_centered: int = 10,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        eps: float = 1e-15,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.steps_before_using_centered: int = steps_before_using_centered
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'centered': centered,
            'weight_decay': weight_decay,
            'foreach': foreach,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'LaProp'

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
                state['exp_avg_lr_1'] = 0.0
                state['exp_avg_lr_2'] = 0.0

                if group['centered']:
                    state['exp_mean_avg_beta2'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    def _get_step_sizes(self, group: ParamGroup, params: list[torch.Tensor]) -> torch.Tensor:
        beta1, beta2 = group['betas']
        lr = group['lr']
        step_sizes = []
        for p in params:
            state = self.state[p]
            state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1.0 - beta1) * lr
            state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1.0 - beta2)
            if isinstance(lr, torch.Tensor):
                bias_correction1 = torch.where(lr != 0.0, state['exp_avg_lr_1'] / lr, 1.0)
            else:
                bias_correction1 = state['exp_avg_lr_1'] / lr if lr != 0.0 else 1.0
            step_sizes.append(-1.0 / bias_correction1)

        if all(isinstance(value, float) for value in step_sizes):
            return torch.tensor(step_sizes, dtype=torch.float32, device=params[0].device)

        return torch.stack([
            torch.as_tensor(value, dtype=torch.float32, device=params[0].device) for value in step_sizes
        ])

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        bias_correction2_sq: float | torch.Tensor,
        step_sizes: torch.Tensor,
        use_centered: bool,
    ) -> None:
        beta1, beta2 = group['betas']
        exp_avgs, exp_avg_sqs = state['exp_avg'], state['exp_avg_sq']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        second_moments = exp_avg_sqs
        if group['centered']:
            means = state['exp_mean_avg_beta2']
            torch._foreach_lerp_(means, grads, weight=1.0 - beta2)
            if use_centered:
                second_moments = torch._foreach_sub(exp_avg_sqs, torch._foreach_pow(means, 2))

        de_noms = self.apply_ams_bound_foreach(
            group['ams_bound'], second_moments, state.get('max_exp_avg_sq', []), group['eps']
        )
        torch._foreach_div_(de_noms, bias_correction2_sq)

        torch._foreach_mul_(exp_avgs, beta1)
        foreach_addcdiv_(exp_avgs, grads, de_noms, value=(1.0 - beta1) * group['lr'])

        torch._foreach_addcmul_(params, exp_avgs, step_sizes.unbind())

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

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

            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            if not group.get('cautious') and self.can_use_foreach(group, group.get('foreach', self.foreach)):
                state_keys = ['exp_avg', 'exp_avg_sq']
                if group['centered']:
                    state_keys.append('exp_mean_avg_beta2')
                if group['ams_bound']:
                    state_keys.append('max_exp_avg_sq')
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=state_keys)
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    step_sizes = self._get_step_sizes(group, batch['params'])
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch, bias_correction2_sq, step_sizes,
                        group['step'] > self.steps_before_using_centered,
                    )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, exp_mean_avg_beta2 = (
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state.get('exp_mean_avg_beta2', None),
                )

                p, grad, exp_avg, exp_avg_sq, exp_mean_avg_beta2 = self.view_as_real(
                    p, grad, exp_avg, exp_avg_sq, exp_mean_avg_beta2
                )

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1.0 - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1.0 - beta2)

                bias_correction1: float = state['exp_avg_lr_1'] / group['lr'] if group['lr'] != 0.0 else 1.0
                step_size: float = 1.0 / bias_correction1

                second_moment = exp_avg_sq
                if group['centered']:
                    exp_mean_avg_beta2.lerp_(grad, weight=1.0 - beta2)
                    if group['step'] > self.steps_before_using_centered:
                        second_moment = exp_avg_sq - exp_mean_avg_beta2.pow(2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=second_moment,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )
                de_nom.div_(bias_correction2_sq)

                exp_avg.mul_(beta1).addcdiv_(grad, de_nom, value=(1.0 - beta1) * group['lr'])

                if group.get('cautious'):
                    update = exp_avg.clone()
                    self.apply_cautious(update, grad)
                else:
                    update = exp_avg

                p.add_(update, alpha=-step_size)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

        return loss
