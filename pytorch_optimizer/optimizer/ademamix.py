import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import group_tensors_by_device_and_dtype


class AdEMAMix(BaseOptimizer):
    """Adam with a mixture of fast and slow gradient momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for fast gradient momentum, squared gradients, and slow gradient momentum.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        alpha: Weight of slow momentum relative to fast momentum.
        t_alpha_beta3: Number of steps to warm up `alpha` and the slow momentum decay. `None` disables warmup.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        alpha: float = 5.0,
        t_alpha_beta3: float | None = None,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
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
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdEMAMix'

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
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: float | None, step: int, alpha: float) -> float:
        return alpha if t_alpha_beta3 is None else min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: float | None, step: int, beta1: float, beta3: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        log_beta1, log_beta3 = math.log(beta1), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        exp_avg_slows: list[torch.Tensor],
        bias_correction1: float,
        bias_correction2_sq: float,
        alpha_t: float,
        beta3_t: float,
    ) -> None:
        beta1, beta2, _ = group['betas']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        torch._foreach_lerp_(exp_avg_slows, grads, weight=1.0 - beta3_t)

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(de_noms, bias_correction2_sq)
        torch._foreach_add_(de_noms, group['eps'])

        if group.get('cautious'):
            updates = [exp_avg.clone() for exp_avg in exp_avgs]
            for update, grad in zip(updates, grads):
                self.apply_cautious(update, grad)
            torch._foreach_div_(updates, bias_correction1)
        else:
            updates = torch._foreach_div(exp_avgs, bias_correction1)

        torch._foreach_add_(updates, exp_avg_slows, alpha=alpha_t)
        torch._foreach_div_(updates, de_noms)

        if group.get('stable_adamw'):
            rms = [self.get_stable_adamw_rms(grad, exp_avg_sq) for grad, exp_avg_sq in zip(grads, exp_avg_sqs)]
            torch._foreach_div_(updates, rms)

        torch._foreach_add_(params, updates, alpha=-group['lr'])

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3)

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq', 'exp_avg_slow']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch['exp_avg'],
                        batch['exp_avg_sq'],
                        batch['exp_avg_slow'],
                        bias_correction1,
                        bias_correction2_sq,
                        alpha_t,
                        beta3_t,
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
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg_slow.lerp_(grad, weight=1.0 - beta3_t)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                update = exp_avg.clone()
                if group.get('cautious'):
                    self.apply_cautious(update, grad)

                update.div_(bias_correction1).add_(exp_avg_slow, alpha=alpha_t).div_(de_nom)

                if group.get('stable_adamw'):
                    update.div_(self.get_stable_adamw_rms(grad, exp_avg_sq))

                p.add_(update, alpha=-group['lr'])

        return loss


class SimplifiedAdEMAMix(BaseOptimizer):
    """Adaptive updates that mix the current gradient with gradient momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for gradient momentum and squared gradients.
        alpha: Coefficient for mixing the current gradient and EMA.
        beta1_warmup: Number of warmup steps used to increase beta1.
        min_beta1: Minimum value of beta1 to start from.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Betas = (0.99, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 0.0,
        beta1_warmup: int | None = None,
        min_beta1: float = 0.9,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(min_beta1, 'min_beta1')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'beta1_warmup': beta1_warmup,
            'min_beta1': min_beta1,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SimplifiedAdEMAMix'

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
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['num_sum'] = 0.0
                state['den_sum'] = 0.0

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:
        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))

        return beta_end

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        beta1: float,
    ) -> None:
        beta2 = group['betas'][1]

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        den_sums: list[float] = []
        for p in params:
            state = self.state[p]
            state['num_sum'] = beta1 * state['num_sum'] + 1.0
            state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)
            den_sums.append(math.sqrt(state['den_sum']))

        de_noms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(de_noms, [den_sum * group['eps'] for den_sum in den_sums])

        updates = torch._foreach_mul(grads, group['alpha'])
        torch._foreach_add_(updates, exp_avgs)
        torch._foreach_div_(updates, de_noms)
        torch._foreach_div_(updates, den_sums)

        torch._foreach_add_(params, updates, alpha=-group['lr'])

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

            if group['beta1_warmup']:
                beta1 = self.linear_hl_warmup_scheduler(
                    group['step'], beta_end=beta1, beta_start=group['min_beta1'], warmup=group['beta1_warmup']
                )

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch['exp_avg'], batch['exp_avg_sq'], beta1
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
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state['num_sum'] = beta1 * state['num_sum'] + 1.0
                state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(math.sqrt(state['den_sum']) * group['eps'])

                update = (group['alpha'] * grad + exp_avg).div_(de_nom).div_(math.sqrt(state['den_sum']))

                p.add_(update, alpha=-group['lr'])

        return loss
