import math
from typing import Literal

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype
from pytorch_optimizer.optimizer.utils.galore import GaLoreProjector, update_unprojected_foreach

SCALE_TYPE = Literal['channel', 'tensor']


class ApolloDQN(BaseOptimizer):
    """Adaptive updates with a diagonal quasi-Newton preconditioner.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        init_lr: Initial learning rate (default lr / 1000).
        beta: Coefficient used for computing running averages of gradient.
        rebound: Rectified bound for diagonal Hessian. Options: 'constant', 'belief'.
        weight_decay: Weight decay coefficient.
        weight_decay_type: Type of weight decay. Options: 'l2', 'decoupled', 'stable'.
        warmup_steps: Number of warmup steps.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates. `None` enables batching for supported CUDA parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        init_lr: float | None = 1e-5,
        beta: float = 0.9,
        rebound: str = 'constant',
        weight_decay: float = 0.0,
        weight_decay_type: str = 'l2',
        warmup_steps: int = 500,
        eps: float = 1e-4,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_options(rebound, 'rebound', ['constant', 'belief'])
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_options(weight_decay_type, 'weight_decay_type', ['l2', 'decoupled', 'stable'])
        self.validate_non_negative(eps, 'eps')

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.init_lr: float = init_lr if init_lr is not None else lr / 1000.0
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'init_lr': self.init_lr,
            'beta': beta,
            'rebound': rebound,
            'weight_decay': weight_decay,
            'weight_decay_type': weight_decay_type,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ApolloDQN'

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
                state['exp_avg_grad'] = torch.zeros_like(p)
                state['approx_hessian'] = torch.zeros_like(p)
                state['update'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        current_lr: float | torch.Tensor,
        alpha: float | torch.Tensor,
        eps: float | torch.Tensor,
    ) -> None:
        averages, hessians, updates = state['exp_avg_grad'], state['approx_hessian'], state['update']

        if self.maximize:
            torch._foreach_neg_(grads)

        weight_decay = group['weight_decay']

        if weight_decay > 0.0 and group['weight_decay_type'] == 'l2':
            torch._foreach_add_(grads, params, alpha=weight_decay)

        delta_grads = torch._foreach_sub(grads, averages)
        rebounds = torch._foreach_norm(delta_grads, ord=float('inf')) if group['rebound'] == 'belief' else None
        foreach_add_(averages, delta_grads, alpha=alpha)

        de_noms = torch._foreach_norm(updates, ord=4)
        torch._foreach_add_(de_noms, eps)
        torch._foreach_div_(updates, de_noms)
        squares = torch._foreach_mul(updates, updates)

        torch._foreach_div_(delta_grads, de_noms)
        torch._foreach_mul_(delta_grads, updates)
        deltas = [delta.sum() for delta in delta_grads]
        torch._foreach_mul_(deltas, -alpha)
        torch._foreach_sub_(deltas, [product.sum() for product in torch._foreach_mul(hessians, squares)])
        torch._foreach_addcmul_(hessians, squares, deltas)

        de_noms = torch._foreach_abs(hessians)

        if rebounds is not None:
            torch._foreach_maximum_(de_noms, rebounds)
            torch._foreach_add_(de_noms, eps / alpha)
        else:
            torch._foreach_clamp_min_(de_noms, 1e-2)

        torch._foreach_copy_(updates, torch._foreach_div(averages, de_noms))

        if weight_decay > 0.0 and group['weight_decay_type'] != 'l2':
            if group['weight_decay_type'] == 'stable':
                decays = [weight_decay / de_nom.mean() for de_nom in de_noms]
                torch._foreach_addcmul_(updates, params, decays)
            else:
                torch._foreach_add_(updates, params, alpha=weight_decay)

        foreach_add_(params, updates, alpha=-current_lr)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            current_lr: float = (
                group['lr']
                if group['step'] >= self.warmup_steps
                else (self.lr - group['init_lr']) * group['step'] / self.warmup_steps + group['init_lr']
            )

            weight_decay, eps = group['weight_decay'], group['eps']

            if group['rebound'] == 'constant':
                eps /= 1e-2

            bias_correction: float = self.debias(group['beta'], group['step'])
            alpha: float = (1.0 - group['beta']) / bias_correction

            if self.can_use_foreach(group, group.get('foreach'), default_on_cuda=True):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, ['exp_avg_grad', 'approx_hessian', 'update']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, current_lr, alpha, eps)

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg_grad, b, d_p = state['exp_avg_grad'], state['approx_hessian'], state['update']

                p, grad, exp_avg_grad, b, d_p = self.view_as_real(p, grad, exp_avg_grad, b, d_p)

                if weight_decay > 0.0 and group['weight_decay_type'] == 'l2':
                    grad.add_(p, alpha=weight_decay)

                delta_grad = grad - exp_avg_grad
                rebound = delta_grad.norm(p=np.inf) if group['rebound'] == 'belief' else 1e-2

                exp_avg_grad.add_(delta_grad, alpha=alpha)

                de_nom = d_p.norm(p=4).add_(eps)
                d_p.div_(de_nom)

                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(de_nom).mul_(d_p).sum().mul(-alpha) - b.mul(v_sq).sum()

                b.addcmul_(v_sq, delta)

                de_nom = b.abs().clamp_(min=rebound)
                if group['rebound'] == 'belief':
                    de_nom.add_(eps / alpha)

                d_p.copy_(exp_avg_grad.div(de_nom))

                if weight_decay > 0.0 and group['weight_decay_type'] != 'l2':
                    decay = weight_decay
                    if group['weight_decay_type'] == 'stable':
                        decay = weight_decay / de_nom.mean().item()

                    d_p.add_(p, alpha=decay)

                p.add_(d_p, alpha=-current_lr)

        return loss


class APOLLO(BaseOptimizer):
    """AdamW with low rank gradient projection and norm based update scaling.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        scale_type: Compute update scaling per `'tensor'` or per `'channel'`.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        correct_bias: Whether to correct bias in Adam.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch unprojected parameter updates. `None` enables batching for supported groups.

    """

    _supports_compiled_foreach = True
    _step_foreach = staticmethod(update_unprojected_foreach)

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-2,
        betas: Betas = (0.9, 0.999),
        scale_type: SCALE_TYPE = 'tensor',
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        correct_bias: bool = True,
        eps: float = 1e-6,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'scale_type': scale_type,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'correct_bias': correct_bias,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'APOLLO'

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

            if ('rank' not in group or p.dim() != 2 or torch.is_complex(p)) and 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            step_size: float = group['lr']
            if group['correct_bias']:
                bias_correction1: float = self.debias(beta1, group['step'])
                bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))
                step_size = step_size * (bias_correction2_sq / bias_correction1)

            use_foreach = self.can_use_foreach(group, group.get('foreach'))

            if use_foreach:
                dense_group = {
                    **group,
                    'params': [p for p in group['params'] if ('rank' not in group or p.dim() != 2)],
                }
                params, grads, state = self.collect_trainable_params(
                    dense_group, self.state, ['exp_avg', 'exp_avg_sq']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch['exp_avg'],
                        batch['exp_avg_sq'],
                        step_size,
                        step_size,
                        self.maximize,
                        group['weight_decouple'],
                        group['fixed_decay'],
                    )

            for p in group['params']:
                if p.grad is None or (use_foreach and ('rank' not in group or p.dim() != 2)):
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if not group['weight_decouple']:
                    self.apply_weight_decay(
                        p, grad, step_size, group['weight_decay'], False, group['fixed_decay']
                    )

                if 'rank' in group and p.dim() == 2 and not torch.is_complex(p):
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            projection_type=group['projection_type'],
                        )

                    grad = state['projector'].project(grad, group['step'], from_random_matrix=True)

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                norm_grad = de_nom
                torch.div(exp_avg, de_nom, out=norm_grad)

                if 'projector' in state:
                    if group['scale_type'] == 'channel':
                        norm_dim: int = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                        scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (torch.norm(grad, dim=norm_dim) + 1e-8)
                        if norm_dim == 1:
                            scaling_factor = scaling_factor.unsqueeze(1)
                    else:
                        scaling_factor = torch.norm(norm_grad) / (torch.norm(grad) + 1e-8)

                    scaling_grad = norm_grad
                    torch.mul(grad, scaling_factor, out=scaling_grad)

                    scaling_grad_norm = torch.norm(scaling_grad)
                    if 'scaling_grad' in state:
                        limiter = (
                            max(
                                scaling_grad_norm / (state['scaling_grad'] + 1e-8),
                                1.01,
                            )
                            / 1.01
                        )

                        scaling_grad.div_(limiter)
                        scaling_grad_norm.div_(limiter)

                    state['scaling_grad'] = scaling_grad_norm

                    norm_grad = scaling_grad.mul_(np.sqrt(group['scale']))
                    norm_grad = state['projector'].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group['weight_decouple']:
                    self.apply_weight_decay(
                        p,
                        grad,
                        lr=step_size,
                        weight_decay=group['weight_decay'],
                        weight_decouple=True,
                        fixed_decay=group['fixed_decay'],
                    )

        return loss
