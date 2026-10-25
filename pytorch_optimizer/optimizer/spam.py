import math
from collections.abc import Callable, Sequence

import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    foreach_addcdiv_,
    foreach_lerp_,
    group_tensors_by_device_and_dtype,
)


class CosineDecay:
    """Cosine decay of a scalar value using PyTorch's `CosineAnnealingLR`.

    Args:
        death_rate: Initial value to be decayed.
        t_max: Maximum number of iterations for the decay.
        eta_min: Minimum value of the parameter after decay. Defaults to 0.
        last_epoch: The index of the last epoch. Defaults to -1.

    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        """Advance the cosine decay scheduler at the given step.

        Args:
            current_step: Current step index.

        """
        self.cosine_stepper.last_epoch = current_step
        self.sgd.step()
        self.cosine_stepper.step()

    def get_death_rate(self, current_step: int) -> float:
        """Get the updated rate (death_rate) at the given step.

        Args:
            current_step: Current step index.

        """
        if current_step >= self.t_max:
            return self.eta_min

        self.step(current_step)

        return self.sgd.param_groups[0]['lr']

    def state_dict(self) -> dict:
        return {'optimizer': self.sgd.state_dict(), 'scheduler': self.cosine_stepper.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.sgd.load_state_dict(state_dict['optimizer'])
        self.cosine_stepper.load_state_dict(state_dict['scheduler'])


class SPAM(BaseOptimizer):
    """Adam with sparse update masks, gradient spike clipping, and momentum resets.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        density: Expected fraction of 2D parameter entries to update between mask resets.
        weight_decay: Weight decay coefficient.
        warmup_epoch: Number of steps to warm up after each momentum reset.
        threshold: Squared gradient to second moment ratio above which to clip spikes.
        grad_accu_steps: Steps after a reset before spike clipping begins.
        update_proj_gap: Number of steps between mask updates and momentum resets.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch FP32/FP64 moments and updates. `None` enables batching on CUDA.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        density: float = 1.0,
        weight_decay: float = 0.0,
        warmup_epoch: int = 50,
        threshold: int = 5000,
        grad_accu_steps: int = 20,
        update_proj_gap: int = 500,
        eps: float = 1e-6,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(warmup_epoch, 'warmup_epoch')
        self.validate_non_negative(density, 'density')
        self.validate_non_negative(threshold, 'threshold')
        self.validate_non_negative(grad_accu_steps, 'grad_accu_steps')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        self.density = density
        self.warmup_epoch = warmup_epoch
        self.threshold = threshold
        self.grad_accu_steps = grad_accu_steps
        self.update_proj_gap = update_proj_gap
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

        self.warmup = CosineDecay(0.99, self.warmup_epoch)

        self.init_masks()

        self.state['total_step'] = 0
        self.state['current_step'] = self.warmup_epoch + 1

    @staticmethod
    def initialize_random_rank_boolean_tensor(m: int, n: int, density: float, device: torch.device) -> torch.Tensor:
        """Create a boolean matrix with an expected fraction of selected entries.

        Args:
            m: Number of rows.
            n: Number of columns.
            density: Probability of selecting each entry. `1` selects all entries.
            device: Device for the matrix.

        Returns:
            torch.Tensor: Boolean mask with shape `(m, n)`.

        """
        total_elements: int = m * n
        non_zero_count: int = int(density * total_elements)

        tensor = torch.zeros(total_elements, dtype=torch.bool, device=device)

        if non_zero_count > 0:
            tensor[torch.randperm(total_elements, device=device)[:non_zero_count]] = True

        return tensor.view(m, n)

    def update_mask_random(self, p: torch.Tensor, old_mask: torch.Tensor) -> torch.Tensor:
        """Resample a parameter mask and retain moments for entries in both masks.

        Args:
            p: Parameter tensor to mask.
            old_mask: Previous boolean mask.

        Returns:
            torch.Tensor: New boolean mask with the configured expected density.

        """
        new_mask: torch.Tensor = torch.rand_like(p) < self.density

        exp_avg = torch.zeros_like(p[new_mask])
        exp_avg_sq = torch.zeros_like(p[new_mask])

        intersection_mask = new_mask & old_mask
        new_intersection_indices = intersection_mask[new_mask]
        old_intersection_indices = intersection_mask[old_mask]

        state = self.state[p]
        exp_avg[new_intersection_indices] = state['exp_avg'][old_intersection_indices]
        exp_avg_sq[new_intersection_indices] = state['exp_avg_sq'][old_intersection_indices]

        state['exp_avg'] = exp_avg
        state['exp_avg_sq'] = exp_avg_sq

        return new_mask

    def update_masks(self) -> None:
        """Resample matrix masks and retain momentum entries shared with the old masks."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.dim() == 2 and 'mask' in state:
                    state['mask'] = self.update_mask_random(p, state['mask'])
                    p.mask = state['mask']

    def init_masks(self) -> None:
        """Initialize sparse update masks for 2D parameters."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.dim() == 2 and 'mask' not in state:
                    state['mask'] = self.initialize_random_rank_boolean_tensor(
                        m=p.shape[0],
                        n=p.shape[1],
                        density=self.density,
                        device=p.device,
                    )

    def __str__(self) -> str:
        return 'SPAM'

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['warmup'] = self.warmup.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        if 'warmup' in state_dict:
            self.warmup.load_state_dict(state_dict['warmup'])

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        clip: bool,
    ) -> Sequence[torch.Tensor]:
        if clip:
            limits = torch._foreach_mul(state['exp_avg_sq'], self.threshold)
            clipped = [
                torch.where(grad.square() > limit, grad.sign() * limit.sqrt(), grad)
                for grad, limit in zip(grads, limits)
            ]
            torch._foreach_copy_(grads, clipped)

        beta1, beta2 = group['betas']
        torch._foreach_lerp_(state['exp_avg'], grads, weight=1.0 - beta1)
        torch._foreach_mul_(state['exp_avg_sq'], beta2)
        torch._foreach_addcmul_(state['exp_avg_sq'], grads, grads, value=1.0 - beta2)

        de_noms = torch._foreach_sqrt(state['exp_avg_sq'])
        torch._foreach_add_(de_noms, group['eps'])

        return de_noms

    def _step_group_foreach(self, group: ParamGroup, step_size: float | torch.Tensor) -> None:
        params, full_grads, _ = self.collect_trainable_params(group, self.state)
        grads = []
        current_step = self.state['total_step'] + 1

        for p, grad in zip(params, full_grads):
            self.maximize_gradient(grad, maximize=self.maximize)
            state = self.state[p]

            if 'mask' in state:
                grad = grad[state['mask']]

            if 'exp_avg' not in state or current_step % self.update_proj_gap == 0:
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)

            grads.append(grad)

        states = {key: [self.state[p][key] for p in params] for key in ('exp_avg', 'exp_avg_sq')}

        clip = self.threshold != 0 and current_step >= self.grad_accu_steps and (
            self.update_proj_gap == 0 or current_step % self.update_proj_gap >= self.grad_accu_steps
        )

        for batch in group_tensors_by_device_and_dtype(params, grads, states):
            de_noms = self._step_foreach(group, batch['params'], batch['grads'], batch, clip)

            unmasked_params, unmasked_avgs, unmasked_de_noms = [], [], []

            for p, exp_avg, de_nom in zip(batch['params'], batch['exp_avg'], de_noms):
                mask = self.state[p].get('mask')

                if mask is None:
                    unmasked_params.append(p)
                    unmasked_avgs.append(exp_avg)
                    unmasked_de_noms.append(de_nom)
                    continue

                update = torch.zeros_like(p.grad)
                update.masked_scatter_(mask, exp_avg / de_nom)
                p.add_(update, alpha=-step_size)

                if group['weight_decay'] > 0.0:
                    p.copy_(torch.where(mask, p * (1.0 - group['lr'] * group['weight_decay']), p))

            if unmasked_params:
                foreach_addcdiv_(unmasked_params, unmasked_avgs, unmasked_de_noms, value=-step_size)
                self.apply_weight_decay_foreach(unmasked_params, [], group['lr'], group['weight_decay'], True, False)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_factor: float = 1.0 - self.warmup.get_death_rate(self.state['current_step'])

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] * bias_correction2_sq / bias_correction1

            if self.can_use_foreach(
                group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
            ):
                self._step_group_foreach(group, step_size * scale_factor)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if 'mask' in state:
                    grad = grad[state['mask']]

                if ('exp_avg' not in state) or (self.state['total_step'] + 1) % self.update_proj_gap == 0:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.threshold != 0:
                    current_step: int = self.state['total_step'] + 1
                    if current_step >= self.grad_accu_steps and (
                        self.update_proj_gap == 0 or current_step % self.update_proj_gap >= self.grad_accu_steps
                    ):
                        mask = grad.pow(2) > (self.threshold * exp_avg_sq)
                        grad[mask] = grad[mask].sign() * torch.sqrt(exp_avg_sq[mask] * self.threshold)

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                if 'mask' in state:
                    grad_full = torch.zeros_like(p.grad)
                    grad_full[state['mask']] = exp_avg / de_nom
                    p.add_(grad_full, alpha=-step_size * scale_factor)
                else:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size * scale_factor)

                decay_param = p[state['mask']] if 'mask' in state else p
                self.apply_weight_decay(
                    decay_param,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )
                if 'mask' in state:
                    p[state['mask']] = decay_param

        self.state['total_step'] += 1
        self.state['current_step'] += 1

        if (self.state['total_step'] != 0) and (self.state['total_step'] + 1) % self.update_proj_gap == 0:
            self.update_masks()
            self.state['current_step'] = 0
            self.warmup = CosineDecay(0.99, self.warmup_epoch)

        return loss


class StableSPAM(BaseOptimizer):
    """Adam with adaptive gradient scaling, spike clipping, and momentum resets.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        gamma1: Decay rate for the gradient norm average. `-1` uses `beta1`.
        gamma2: Decay rate for the squared gradient norm average.
        theta: Decay rate for the maximum absolute gradient average.
        t_max: Steps for cosine decay of momentum coefficients. `None` disables decay.
        eta_min: Minimum multiplier for the cosine decayed momentum coefficients.
        weight_decay: Weight decay coefficient.
        update_proj_gap: Steps between momentum resets.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them when supported. Defaults to `False`.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        gamma1: float = 0.7,
        gamma2: float = 0.9,
        theta: float = 0.999,
        t_max: int | None = None,
        eta_min: float = 0.5,
        weight_decay: float = 0.0,
        update_proj_gap: int = 1000,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        self.gamma1: float = betas[0] if gamma1 == -1.0 else gamma1
        self.gamma2: float = gamma2
        self.theta: float = theta
        self.t_max = t_max
        self.update_proj_gap = update_proj_gap
        self.warmup = CosineDecay(1.0, t_max, eta_min=eta_min) if t_max is not None else None
        self.maximize = maximize

        self.total_step: int = 0
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableSPAM'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['total_step'] = self.total_step
        if self.warmup is not None:
            state['warmup'] = self.warmup.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.total_step = state_dict.get('total_step', 0)
        if self.warmup is not None and 'warmup' in state_dict:
            self.warmup.load_state_dict(state_dict['warmup'])

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

            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)
                state['m_norm_t'] = torch.zeros((), device=grad.device, dtype=grad.dtype)
                state['v_norm_t'] = torch.zeros((), device=grad.device, dtype=grad.dtype)
                state['m_max_t'] = torch.zeros((), device=grad.device, dtype=grad.dtype)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        beta1: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        theta_t: float,
        norm_weight: float,
        norm_correction1: float,
        norm_correction2: float,
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(params, grads, group['lr'], group['weight_decay'], True, False)

        updates = []

        for index, grad in enumerate(grads):
            m_max = state['m_max_t'][index]
            max_grad = grad.abs().max()
            m_max.lerp_(max_grad, weight=1.0 - self.theta)
            m_max_hat = m_max / theta_t

            clipped = torch.where(grad.abs() > m_max_hat, grad / max_grad * m_max_hat, grad)
            grad_norm = torch.linalg.norm(clipped)
            active = grad_norm > 0

            m_norm, v_norm = state['m_norm_t'][index], state['v_norm_t'][index]
            new_m_norm = m_norm.clone()
            foreach_lerp_([new_m_norm], [grad_norm.reshape_as(m_norm)], weight=norm_weight)
            m_norm.copy_(torch.where(active, new_m_norm, m_norm))
            v_norm.copy_(torch.where(active, v_norm.lerp(grad_norm.square(), weight=1.0 - self.gamma2), v_norm))

            c_norm = (m_norm / norm_correction1) / ((v_norm / norm_correction2).sqrt().add_(group['eps']))
            grad.copy_(torch.where(active, clipped / torch.where(active, grad_norm, 1.0) * c_norm, clipped))

            exp_avg, exp_avg_sq = state['exp_avg'][index], state['exp_avg_sq'][index]
            new_exp_avg = exp_avg.clone()
            foreach_lerp_([new_exp_avg], [grad], weight=1.0 - beta1)
            exp_avg.copy_(torch.where(active, new_exp_avg, exp_avg))
            new_exp_avg_sq = exp_avg_sq.mul(group['betas'][1]).addcmul_(grad, grad, value=1.0 - group['betas'][1])
            exp_avg_sq.copy_(torch.where(active, new_exp_avg_sq, exp_avg_sq))

            de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])
            updates.append(torch.where(active, exp_avg / de_nom, 0.0))

        foreach_add_(params, updates, alpha=-group['lr'] / bias_correction1)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_step += 1

        scale: float = self.warmup.get_death_rate(self.total_step) if self.warmup is not None else 1.0

        for group in self.param_groups:
            self.init_group(group)
            if self.total_step % self.update_proj_gap == 0:
                group['step'] = 1
                for p in group['params']:
                    if 'exp_avg' in self.state[p]:
                        self.state[p]['exp_avg'].zero_()
                        self.state[p]['exp_avg_sq'].zero_()
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']
            beta1 *= scale

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction2_sq: float = math.sqrt(bias_correction2)

            step_size: float = group['lr'] / bias_correction1

            theta_t: float = 1.0 - self.theta**self.total_step

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, ['exp_avg', 'exp_avg_sq', 'm_norm_t', 'v_norm_t', 'm_max_t']
                )

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = self._step_foreach
                    if self._compiled_step_foreach is not None and batch['params'][0].dtype != torch.float16:
                        update = self._compiled_step_foreach

                    update(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch,
                        beta1,
                        bias_correction1,
                        bias_correction2_sq,
                        theta_t,
                        1.0 - self.gamma1 * scale,
                        1.0 - (self.gamma1 * scale) ** self.total_step,
                        1.0 - self.gamma2**self.total_step,
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                max_grad = torch.max(grad.abs())

                exp_avg, exp_avg_sq, m_max_t = state['exp_avg'], state['exp_avg_sq'], state['m_max_t']

                m_max_t.lerp_(max_grad, weight=1.0 - self.theta)

                m_max_hat = m_max_t / theta_t

                mask = grad.abs() > m_max_hat
                if mask.sum() > 0:
                    grad[mask] = grad[mask] / max_grad * m_max_hat

                grad_norm = torch.linalg.norm(grad)
                if grad_norm == 0:
                    continue

                m_norm_t, v_norm_t = state['m_norm_t'], state['v_norm_t']
                m_norm_t.lerp_(grad_norm, weight=1.0 - self.gamma1 * scale)
                v_norm_t.lerp_(grad_norm.pow(2), weight=1.0 - self.gamma2)

                m_norm_hat = m_norm_t / (1.0 - (self.gamma1 * scale) ** self.total_step)
                v_norm_hat = v_norm_t / (1.0 - self.gamma2**self.total_step)

                c_norm_t = m_norm_hat.div_(v_norm_hat.sqrt_().add_(group['eps']))

                grad.div_(grad_norm).mul_(c_norm_t)

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
