import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdaBelief(BaseOptimizer):
    """Adaptive updates based on gradient prediction error.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient mean and squared gradient prediction error.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        rectify: Perform the rectified update similar to RAdam.
        n_sma_threshold: Minimum effective simple moving average length for rectification.
        degenerated_to_sgd: Use an SGD update before the moving average reaches the rectification threshold.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

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
        foreach: bool | None = None,
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
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_vars: list[torch.Tensor],
        step_size: float,
    ) -> None:
        beta1, beta2 = group['betas']
        lr = group['lr']

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
        )

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)

        grad_residuals = torch._foreach_sub(grads, exp_avgs)

        torch._foreach_mul_(exp_avg_vars, beta2)
        torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1.0 - beta2)
        torch._foreach_add_(exp_avg_vars, group['eps'])

        de_noms = torch._foreach_sqrt(exp_avg_vars)
        torch._foreach_div_(de_noms, bias_correction2_sq)
        torch._foreach_add_(de_noms, group['eps'])

        torch._foreach_addcdiv_(params, exp_avgs, de_noms, value=-step_size)

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

            exp_avg.lerp_(s_grad, weight=1.0 - beta1)

            grad_residual = grad - exp_avg
            exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2).add_(group['eps'])

            de_nom = self.apply_ams_bound(
                ams_bound=group['ams_bound'],
                exp_avg_sq=exp_avg_var,
                max_exp_avg_sq=state.get('max_exp_avg_var', None),
                eps=0.0,
                exp_avg_sq_eps=0.0,
            )

            if not group['rectify']:
                de_nom.div_(bias_correction2_sq).add_(group['eps'])
                p.addcdiv_(exp_avg, de_nom, value=-step_size)
                continue

            de_nom.add_(group['eps'])

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
                    self._step_foreach(
                        group, params, grads, state_dict['exp_avg'], state_dict['exp_avg_var'], step_size
                    )
            else:
                self._step_per_param(group, step_size, n_sma)

        return loss
