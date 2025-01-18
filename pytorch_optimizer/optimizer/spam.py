import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class CosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using PyTorch's built-in `CosineAnnealingLR`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    :param last_epoch: Optional[int]. the index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd = torch.optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.T_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        r"""One step of the cosine decay scheduler.

        :param current_step: int. Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_death_rate(self, current_step: int) -> float:
        r"""Get the updated rate (death_rate) at the given step.

        :param current_step: int. Current step index.
        """
        if current_step >= self.T_max:
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
        warmup_epoch: int = 150,
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
        self.validate_non_negative(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        self.density = density
        self.warmup_epoch = warmup_epoch
        self.threshold = threshold
        self.grad_accu_steps = grad_accu_steps
        self.update_proj_gap = update_proj_gap
        self.warmup = CosineDecay(0.99, warmup_epoch)

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
            **kwargs,
        }
        super().__init__(params, defaults)

        self.init_masks()

        self.state['total_step'] = 0
        self.state['current_step'] = warmup_epoch + 1

    @staticmethod
    def initialize_random_rank_boolean_tensor(m: int, n: int, density: float) -> torch.Tensor:
        r"""Create an (m x n) boolean tensor with `density` fraction of True entries.

        :param m: int. number of rows.
        :param n: int. number of columns.
        :param density: float. fraction of True entries. 1.0 means all True.
        """
        total_elements: int = m * n
        non_zero_count: int = int(density * total_elements)

        tensor = torch.zeros((m, n), dtype=torch.bool)

        if non_zero_count == 0:
            return tensor

        indices = torch.randperm(total_elements)[:non_zero_count]
        rows, cols = indices // n, indices % n
        tensor[rows, cols] = True

        return tensor

    def update_mask_random(self, density: float, p: torch.Tensor, old_mask: torch.Tensor) -> torch.Tensor:
        r"""Update a random mask.

        Create a new random mask with the same density, compute overlap ratio with old_mask, and update the EMA for
        the overlap region.

        :param density: float. fraction of elements to keep.
        :param p: torch.Tensor. parameter to which the mask is applied.
        :param old_mask: torch.Tensor. previous binary mask.
        """
        new_mask: torch.Tensor = torch.rand_like(p) < density

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
                if 'mask' in state:
                    new_mask = self.update_mask_random(self.density, p, state['mask'])
                    state['mask'] = new_mask
                    p.mask = new_mask

    def init_masks(self) -> None:
        r"""Initialize random masks for each parameter group that has 'density'."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.dim() == 2 and 'mask' not in state:
                    state['mask'] = self.initialize_random_rank_boolean_tensor(
                        p.shape[0],
                        p.shape[1],
                        density=self.density,
                    ).to(p.device)

    def __str__(self) -> str:
        return 'SPAM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                if (self.state['total_step'] + 1) % self.update_proj_gap == 0:
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
