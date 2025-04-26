import math
from typing import Optional

import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class CosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using PyTorch's built-in `CosineAnnealingLR`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    :param last_epoch: Optional[int]. the index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        r"""One step of the cosine decay scheduler.

        :param current_step: int. Current step index.
        """
        self.cosine_stepper.last_epoch = current_step
        self.cosine_stepper.step()

    def get_death_rate(self, current_step: int) -> float:
        r"""Get the updated rate (death_rate) at the given step.

        :param current_step: int. Current step index.
        """
        if current_step >= self.t_max:
            return self.eta_min

        self.step(current_step)

        return self.sgd.param_groups[0]['lr']


class SPAM(BaseOptimizer):
    r"""Spike-Aware Adam with Momentum Reset for Stable LLM Training.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param density: float. density parameter. only used for 2d parameters (e.g. Linear).
    :param weight_decay: float. weight decay (L2 penalty).
    :param warmup_epoch: int: number of epochs to warm up. defaults to 50.
    :param threshold: int. threshold for gradient masking. defaults to 5000.
    :param grad_accu_steps: int. gradient accumulation steps before threshold-based masking applies. defaults to 20.
    :param update_proj_gap: int. update projection gap.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        density: float = 1.0,
        weight_decay: float = 0.0,
        warmup_epoch: int = 50,
        threshold: int = 5000,
        grad_accu_steps: int = 20,
        update_proj_gap: int = 500,
        eps: float = 1e-6,
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

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps, **kwargs}
        super().__init__(params, defaults)

        self.warmup = CosineDecay(0.99, self.warmup_epoch)

        self.init_masks()

        self.state['total_step'] = 0
        self.state['current_step'] = self.warmup_epoch + 1

    @staticmethod
    def initialize_random_rank_boolean_tensor(m: int, n: int, density: float, device: torch.device) -> torch.Tensor:
        r"""Create an (m x n) boolean tensor with `density` fraction of True entries.

        :param m: int. number of rows.
        :param n: int. number of columns.
        :param density: float. fraction of True entries. 1.0 means all True.
        :param device: torch.device. device.
        """
        total_elements: int = m * n
        non_zero_count: int = int(density * total_elements)

        tensor = torch.zeros(total_elements, dtype=torch.bool, device=device)

        if non_zero_count > 0:
            tensor[torch.randperm(total_elements, device=device)[:non_zero_count]] = True

        return tensor.view(m, n)

    def update_mask_random(self, p: torch.Tensor, old_mask: torch.Tensor) -> torch.Tensor:
        r"""Update a random mask.

        Create a new random mask with the same density, compute overlap ratio with old_mask, and update the EMA for
        the overlap region.

        :param p: torch.Tensor. parameter to which the mask is applied.
        :param old_mask: torch.Tensor. previous binary mask.
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
        r"""Update masks in each parameter group that has 'density'.

        The new mask is selected randomly, and the overlap ratio with the old mask is printed.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.dim() == 2 and 'mask' in state:
                    state['mask'] = self.update_mask_random(p, state['mask'])
                    p.mask = state['mask']

    def init_masks(self) -> None:
        r"""Initialize random masks for each parameter group that has 'density'."""
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

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_factor: float = 1.0 - self.warmup.get_death_rate(self.state['current_step'])

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] * bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

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
                        grad[mask].sign_().mul_(torch.sqrt(exp_avg_sq[mask] * self.threshold))

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                if 'mask' in state:
                    grad_full = torch.zeros_like(p.grad)
                    grad_full[state['mask']] = exp_avg / de_nom
                    p.add_(grad_full, alpha=-step_size * scale_factor)
                else:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size * scale_factor)

                self.apply_weight_decay(
                    p[state['mask']] if 'mask' in state else p,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

        self.state['total_step'] += 1
        self.state['current_step'] += 1

        if (self.state['total_step'] != 0) and (self.state['total_step'] + 1) % self.update_proj_gap == 0:
            self.update_masks()
            self.state['current_step'] = 0
            self.warmup = CosineDecay(0.99, self.warmup_epoch)

        return loss


class StableSPAM(BaseOptimizer):
    r"""How to Train in 4-Bit More Stably than 16-Bit Adam.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param gamma1: float. gamma1 parameter.
    :param gamma2: float. gamma2 parameter.
    :param theta: float. theta parameter.
    :param t_max: Optional[int]. total number of steps.
    :param eta_min: float. eta_min of CosineDecay.
    :param weight_decay: float. weight decay (L2 penalty).
    :param update_proj_gap: int. update projection gap.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        gamma1: float = 0.7,
        gamma2: float = 0.9,
        theta: float = 0.999,
        t_max: Optional[int] = None,
        eta_min: float = 0.5,
        weight_decay: float = 0.0,
        update_proj_gap: int = 1000,
        eps: float = 1e-8,
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

        self.total_step: int = 0

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps, **kwargs}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableSPAM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['m_norm_t'] = torch.zeros(1, device=p.device, dtype=p.dtype)
                state['v_norm_t'] = torch.zeros(1, device=p.device, dtype=p.dtype)
                state['m_max_t'] = torch.zeros(1, device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_step += 1
        scale: float = self.warmup.get_death_rate(self.total_step) if self.warmup is not None else 1.0

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']
            beta1 *= scale

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction2_sq: float = math.sqrt(bias_correction2)

            step_size: float = group['lr'] / bias_correction1

            theta_t: float = 1.0 - self.theta ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['m_norm_t'] = torch.zeros(1, device=grad.device, dtype=grad.dtype)
                    state['v_norm_t'] = torch.zeros(1, device=grad.device, dtype=grad.dtype)
                    state['m_max_t'] = torch.zeros(1, device=grad.device, dtype=grad.dtype)

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
                    grad[mask].div_(max_grad).mul_(m_max_hat)

                grad_norm = torch.norm(grad)

                m_norm_t, v_norm_t = state['m_norm_t'], state['v_norm_t']
                m_norm_t.lerp_(grad_norm, weight=1.0 - self.gamma1 * scale)
                v_norm_t.lerp_(grad_norm.pow(2), weight=1.0 - self.gamma2)

                m_norm_hat = m_norm_t / (1.0 - (self.gamma1 * scale) ** group['step'])
                v_norm_hat = v_norm_t / (1.0 - self.gamma2 ** group['step'])

                c_norm_t = m_norm_hat.div_(v_norm_hat.sqrt_().add_(group['eps']))

                grad.div_(grad_norm).mul_(c_norm_t)

                if self.update_proj_gap > 0 and self.total_step % self.update_proj_gap == 0:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    group['step'] = 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
