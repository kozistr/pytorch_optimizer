import math
from typing import List

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class ScheduleFreeSGD(BaseOptimizer):
    r"""Schedule-Free SGD.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor, must be between 0 and 1 exclusive.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param r: float. use polynomial weighting in the average with power r.
    :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
        set to 0 for no weighting.
    :param warmup_steps: int. enables a linear learning rate warmup.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'warmup_steps': warmup_steps,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'ScheduleFreeSGD'

    def eval(self):
        for group in self.param_groups:
            momentum = group['momentum']
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - 1.0 / momentum)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            momentum = group['momentum']
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - momentum)
                group['train_mode'] = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            warmup_steps: int = group['warmup_steps']
            schedule: float = group['step'] / warmup_steps if group['step'] < warmup_steps else 1.0

            momentum = group['momentum']

            lr: float = group['lr'] * schedule
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['z'] = p.clone()

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                z = state['z']

                p.lerp_(z, weight=checkpoint)
                p.add_(grad, alpha=lr * (momentum * (1.0 - checkpoint) - 1))

                z.sub_(grad, alpha=lr)

        return loss


class ScheduleFreeAdamW(BaseOptimizer):
    r"""Schedule-Free AdamW.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param r: float. use polynomial weighting in the average with power r.
    :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
        set to 0 for no weighting.
    :param warmup_steps: int. enables a linear learning rate warmup.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2.5e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        ams_bound: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'warmup_steps': warmup_steps,
            'ams_bound': ams_bound,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'use_palm': kwargs.get('use_palm', False),
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'ScheduleFreeAdamW'

    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - 1.0 / beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            warmup_steps: int = group['warmup_steps']
            schedule: float = group['step'] / warmup_steps if group['step'] < warmup_steps else 1.0

            beta1, beta2 = group['betas']

            bias_correction2_sq: float = math.sqrt(1.0 - beta2 ** group['step'])

            lr: float = group['lr'] * schedule * bias_correction2_sq
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            if group['use_palm']:
                beta2: float = 1.0 - group['step'] ** -0.8
                debias: float = (1.0 - beta2) / (1.0 - beta2 ** group['step'])
            else:
                debias: float = beta2

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                z, exp_avg_sq = state['z'], state['exp_avg_sq']
                exp_avg_sq.mul_(debias).addcmul_(grad, grad, value=1.0 - debias)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )

                grad.div_(de_nom)

                p.lerp_(z, weight=checkpoint)
                p.add_(grad, alpha=lr * (beta1 * (1.0 - checkpoint) - 1))

                z.sub_(grad, alpha=lr)

        return loss


class ScheduleFreeRAdam(BaseOptimizer):
    r"""Schedule-Free RAdam.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param degenerated_to_sgd: float. degenerated to SGD.
    :param r: float. use polynomial weighting in the average with power r.
    :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
        set to 0 for no weighting.
    :param warmup_steps: int. enables a linear learning rate warmup.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2.5e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        degenerated_to_sgd: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'degenerated_to_sgd': degenerated_to_sgd,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'use_palm': kwargs.get('use_palm', False),
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ScheduleFreeRAdam'

    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - 1.0 / beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            bias_correction2_sq: float = math.sqrt(1.0 - beta2 ** group['step'])

            lr, n_sma = self.get_rectify_step_size(
                is_rectify=True,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=4,
                degenerated_to_sgd=group['degenerated_to_sgd'],
            )

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1.0)

            if group['use_palm']:
                beta2: float = 1.0 - group['step'] ** -0.8
                debias: float = (1.0 - beta2) / (1.0 - beta2 ** group['step'])
            else:
                debias: float = beta2

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, exp_avg_sq = state['z'], state['exp_avg_sq']
                exp_avg_sq.mul_(debias).addcmul_(grad, grad, value=1.0 - debias)

                if n_sma > 4.0:
                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])
                    grad.div_(de_nom)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.lerp_(z, weight=checkpoint)
                p.add_(grad, alpha=adaptive_y_lr)

                z.sub_(grad, alpha=lr)

        return loss
