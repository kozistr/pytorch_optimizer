import math
from collections.abc import Callable

import torch
from torch.nn.functional import softplus

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_addcdiv_,
    group_tensors_by_device_and_dtype,
)
from pytorch_optimizer.optimizer.utils.gradient import normalize_gradient, unit_norm


class Ranger21(BaseOptimizer):
    """AdamW with positive negative momentum, gradient clipping, and Lookahead.

    Includes gradient centralization and normalization, stable weight decay, norm loss,
    softplus smoothing, and an optional learning rate schedule with warmup and warmdown.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        num_iterations: Total training steps for the built in learning rate schedule.
        lr: Learning rate.
        beta0: Manages the amplitude of the noise introduced by positive negative momentum.
        betas: Decay rates for the first and second moments.
        use_softplus: Use softplus to smooth.
        beta_softplus: Beta parameter for softplus smoothing.
        disable_lr_scheduler: Whether to disable learning rate schedule.
        num_warm_up_iterations: Number of warmup iterations. Ranger21 performs linear learning rate warmup.
        num_warm_down_iterations: Number of warmdown iterations. Ranger21 performs Explore-exploit learning rate
            scheduling.
        warm_down_min_lr: Learning rate at the end of warmdown.
        agc_clipping_value: Maximum gradient-to-parameter norm ratio for adaptive clipping.
        agc_eps: Lower bound for the parameter norm in adaptive clipping.
        centralize_gradients: Use GC both convolution & fc layers.
        normalize_gradients: Use gradient normalization.
        lookahead_merge_time: Steps between Lookahead slow weight updates.
        lookahead_blending_alpha: Interpolation factor from slow weights toward fast weights.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        norm_loss_factor: Coefficient for the unit norm regularization update.
        eps: Term added to the denominator to improve numerical stability.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    _supports_compiled_foreach = True

    def __init__(  # pylint: disable=R0913
        self,
        params: ParamsT,
        num_iterations: int,
        lr: float = 1e-3,
        beta0: float = 0.9,
        betas: Betas = (0.9, 0.999),
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        disable_lr_scheduler: bool = False,
        num_warm_up_iterations: int | None = None,
        num_warm_down_iterations: int | None = None,
        warm_down_min_lr: float = 3e-5,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
        centralize_gradients: bool = True,
        normalize_gradients: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        weight_decay: float = 1e-4,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        norm_loss_factor: float = 1e-4,
        eps: float = 1e-8,
        maximize: bool = False,
        *,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(warm_down_min_lr)
        self.validate_betas(betas)
        self.validate_range(beta0, 'beta0', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(agc_clipping_value, 'agc_clipping_value')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(agc_eps, 'agc_eps')

        self.min_lr = warm_down_min_lr
        self.beta0 = beta0
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.disable_lr_scheduler = disable_lr_scheduler
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.norm_loss_factor = norm_loss_factor
        self.maximize = maximize
        self.foreach = foreach
        self._compiled_preprocess: Callable | None = None

        self.lookahead_step: int = 0
        self.starting_lr: float = lr
        self.current_lr: float = lr

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

        self.num_warm_up_iterations: int = (
            self.build_warm_up_iterations(num_iterations, betas[1])
            if num_warm_up_iterations is None
            else num_warm_up_iterations
        )
        self.num_warm_down_iterations: int = (
            self.build_warm_down_iterations(num_iterations)
            if num_warm_down_iterations is None
            else num_warm_down_iterations
        )
        self.start_warm_down: int = num_iterations - self.num_warm_down_iterations
        self.warm_down_lr_delta: float = self.starting_lr - self.min_lr

    def __str__(self) -> str:
        return 'Ranger21'

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['lookahead_step'] = self.lookahead_step
        state['starting_lr'] = self.starting_lr
        state['current_lr'] = self.current_lr
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.lookahead_step = state_dict.get('lookahead_step', 0)
        self.starting_lr = state_dict.get('starting_lr', self.starting_lr)
        self.current_lr = state_dict.get('current_lr', self.current_lr)

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
                state['grad_ma'] = torch.zeros_like(p)
                state['variance_ma'] = torch.zeros_like(p)
                state['lookahead_params'] = p.clone()
                state['neg_grad_ma'] = torch.zeros_like(p)
                state['max_variance_ma'] = torch.zeros_like(p)

    @staticmethod
    def build_warm_up_iterations(total_iterations: int, beta2: float, warm_up_pct: float = 0.22) -> int:
        warm_up_iterations: int = math.ceil(2.0 / (1.0 - beta2))  # default un-tuned linear warmup
        beta_pct: float = warm_up_iterations / total_iterations
        return int(warm_up_pct * total_iterations) if beta_pct > 0.45 else warm_up_iterations

    @staticmethod
    def build_warm_down_iterations(total_iterations: int, warm_down_pct: float = 0.72) -> int:
        start_warm_down: int = int(warm_down_pct * total_iterations)
        return total_iterations - start_warm_down

    def warm_up_dampening(self, lr: float, step: int) -> float:
        if step > self.num_warm_up_iterations:
            return lr

        warm_up_current_pct: float = min(1.0, (step / self.num_warm_up_iterations))

        self.current_lr = lr * warm_up_current_pct

        return self.current_lr

    def warm_down(self, lr: float, iteration: int) -> float:
        if iteration < self.start_warm_down:
            return lr

        # start iteration from 1, not 0
        warm_down_iteration: int = max((iteration + 1) - self.start_warm_down, 1)
        warm_down_pct: float = min(warm_down_iteration / (self.num_warm_down_iterations + 1), 1.0)

        self.current_lr = max(self.starting_lr - self.warm_down_lr_delta * warm_down_pct, self.min_lr)

        return self.current_lr

    def _preprocess_gradient(self, grad: torch.Tensor) -> None:
        if self.centralize_gradients:
            centralize_gradient(grad, gc_conv_only=False)
        if self.normalize_gradients:
            normalize_gradient(grad)

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        super()._compile_foreach(compile_kwargs)
        self._compiled_preprocess = compile_foreach_step(self._preprocess_foreach, compile_kwargs)

    def _preprocess_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        variances: list[torch.Tensor],
        bias_correction2: float | torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.maximize:
            torch._foreach_neg_(grads)

        for p, grad in zip(params, grads):
            grad.copy_(agc(p, grad, self.agc_eps, self.agc_clipping_value))
            self._preprocess_gradient(grad)

        beta2 = group['betas'][1]
        torch._foreach_mul_(variances, beta2)
        torch._foreach_addcmul_(variances, grads, grads, value=1.0 - beta2)
        return [(variance / bias_correction2).sum() for variance in variances]

    def _preprocess_group(self, group: ParamGroup) -> tuple[int, list[torch.Tensor]]:
        bias_correction2 = self.debias(group['betas'][1], group['step'])
        params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=['variance_ma'])

        if self.can_use_foreach(group, group.get('foreach')):
            sums = list(state_dict['variance_ma'])
            preprocess = self._compiled_preprocess or self._preprocess_foreach
            for tensors in group_tensors_by_device_and_dtype(params, grads, state_dict):
                values = preprocess(
                    group, tensors['params'], tensors['grads'], tensors['variance_ma'], bias_correction2
                )
                for index, value in zip(tensors['indices'], values):
                    sums[index] = value
        else:
            sums = []
            for p, grad, variance in zip(params, grads, state_dict['variance_ma']):
                self.maximize_gradient(grad, maximize=self.maximize)
                grad.copy_(agc(p, grad, self.agc_eps, self.agc_clipping_value))
                self._preprocess_gradient(grad)
                variance.mul_(group['betas'][1]).addcmul_(grad, grad, value=1.0 - group['betas'][1])
                sums.append((variance / bias_correction2).sum())

        return sum(p.numel() for p in params), sums

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        lr: float | torch.Tensor,
        step_size: float | torch.Tensor,
        bias_correction2_sq: float | torch.Tensor,
        variance_normalized: float | torch.Tensor,
    ) -> None:
        if group['weight_decouple'] and group['weight_decay'] != 0.0:
            factor = 1.0 - group['weight_decay'] * (1.0 if group['fixed_decay'] else lr) * (1.0 / variance_normalized)
            torch._foreach_mul_(params, factor)

        for p in params:
            correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p).add_(group['eps']))
            p.mul_(1.0 - lr * correction)

        grad_mas, neg_grad_mas = state_dict['grad_ma'], state_dict['neg_grad_ma']
        torch._foreach_maximum_(state_dict['max_variance_ma'], state_dict['variance_ma'])
        de_noms = torch._foreach_sqrt(state_dict['max_variance_ma'])
        torch._foreach_div_(de_noms, bias_correction2_sq)
        torch._foreach_add_(de_noms, group['eps'])
        if self.use_softplus:
            de_noms = [softplus(de_nom, beta=self.beta_softplus) for de_nom in de_noms]

        for grad in grads:
            self._preprocess_gradient(grad)

        torch._foreach_lerp_(grad_mas, grads, weight=1.0 - group['betas'][0] ** 2)
        pn_momentums = torch._foreach_mul(grad_mas, 1.0 + self.beta0)
        torch._foreach_add_(pn_momentums, neg_grad_mas, alpha=-self.beta0)
        noise_norm = math.sqrt((1.0 + self.beta0) ** 2 + self.beta0**2)
        torch._foreach_mul_(pn_momentums, 1.0 / noise_norm)
        foreach_addcdiv_(params, pn_momentums, de_noms, value=-step_size)

    def _step_per_param(
        self, group: ParamGroup, lr: float, step_size: float, bias_correction2_sq: float, variance_normalized: float
    ) -> None:
        noise_norm = math.sqrt((1.0 + self.beta0) ** 2 + self.beta0**2)
        for p in group['params']:
            if p.grad is None:
                continue

            self.apply_weight_decay(
                p,
                None,
                lr,
                group['weight_decay'],
                group['weight_decouple'],
                group['fixed_decay'],
                ratio=1.0 / variance_normalized,
            )
            correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p).add_(group['eps']))
            p.mul_(1.0 - lr * correction)

            state = self.state[p]
            if group['step'] % 2 == 1:
                grad_ma, neg_grad_ma = state['grad_ma'], state['neg_grad_ma']
            else:
                grad_ma, neg_grad_ma = state['neg_grad_ma'], state['grad_ma']

            torch.maximum(state['max_variance_ma'], state['variance_ma'], out=state['max_variance_ma'])
            de_nom = state['max_variance_ma'].sqrt().div_(bias_correction2_sq).add_(group['eps'])
            if self.use_softplus:
                de_nom = softplus(de_nom, beta=self.beta_softplus)

            grad = p.grad
            self._preprocess_gradient(grad)
            grad_ma.lerp_(grad, weight=1.0 - group['betas'][0] ** 2)
            pn_momentum = grad_ma.mul(1.0 + self.beta0).add_(neg_grad_ma, alpha=-self.beta0)
            pn_momentum.mul_(1.0 / noise_norm)
            p.addcdiv_(pn_momentum, de_nom, value=-step_size)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        variance_ma_sum: float | torch.Tensor = 1.0

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            group_size, variance_sums = self._preprocess_group(group)
            param_size += group_size
            for variance_sum in variance_sums:
                variance_ma_sum += variance_sum

        if param_size == 0:
            raise ZeroParameterSizeError

        variance_normalized = math.sqrt(variance_ma_sum / param_size)

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            if self.disable_lr_scheduler:
                lr: float = group['lr']
            else:
                lr: float = self.warm_up_dampening(group['lr'], group['step'])
                lr = self.warm_down(lr, group['step'])

            step_size: float = self.apply_adam_debias(group.get('adam_debias', False), lr, bias_correction1)

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['grad_ma', 'neg_grad_ma', 'variance_ma', 'max_variance_ma']
                )
                if group['step'] % 2 == 0:
                    state_dict['grad_ma'], state_dict['neg_grad_ma'] = state_dict['neg_grad_ma'], state_dict['grad_ma']
                for tensors in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group,
                        tensors['params'],
                        tensors['grads'],
                        tensors,
                        lr,
                        step_size,
                        bias_correction2_sq,
                        variance_normalized,
                    )
            else:
                self._step_per_param(group, lr, step_size, bias_correction2_sq, variance_normalized)

        self.lookahead_process_step()

        return loss

    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                if self.can_use_foreach(group, group.get('foreach')):
                    params, _, state_dict = self.collect_trainable_params(
                        group, self.state, state_keys=['lookahead_params']
                    )
                    torch._foreach_lerp_(
                        params, state_dict['lookahead_params'], weight=1.0 - self.lookahead_blending_alpha
                    )
                    torch._foreach_copy_(state_dict['lookahead_params'], params)
                    continue
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p.lerp_(state['lookahead_params'], weight=1.0 - self.lookahead_blending_alpha)
                    state['lookahead_params'].copy_(p)
