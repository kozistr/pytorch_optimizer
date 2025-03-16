from collections import defaultdict
from typing import Callable, Dict, List

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, OPTIMIZER_INSTANCE_OR_CLASS, PARAMETERS, STATE


class ScheduleFreeSGD(BaseOptimizer):
    r"""Schedule-Free SGD.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor, must be between 0 and 1 exclusive.
    :param weight_decay: float. weight decay (L2 penalty).
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

            weight: float = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
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
                    weight_decouple=False,
                    fixed_decay=False,
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

            bias_correction2: float = self.debias(beta2, group['step'])

            lr: float = group['lr'] * schedule
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight: float = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
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
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, exp_avg_sq = state['z'], state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq.div(bias_correction2),
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )

                grad.div_(de_nom)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=False,
                    fixed_decay=False,
                )

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
    :param r: float. use polynomial weighting in the average with power r.
    :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
        set to 0 for no weighting.
    :param silent_sgd_phase: bool. the optimizer will not use the first SGD phase of RAdam. This means that the
        optimizer will not update model parameters during the early training steps (e.g., < 5 when β_2 = 0.999), but
        just update the momentum values of the optimizer. This helps stabilize training by ensuring smoother warmup
        behavior and more reliable calculation of the moving average coefficient (`ckp1`). Recommended to set to True.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2.5e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        silent_sgd_phase: bool = True,
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
            'silent_sgd_phase': silent_sgd_phase,
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

            bias_correction2: float = self.debias(beta2, group['step'])

            lr, n_sma = self.get_rectify_step_size(
                is_rectify=True,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=4,
                degenerated_to_sgd=False,
            )
            if lr < 0.0:
                lr = float(not group['silent_sgd_phase'])

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight: float = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1.0)

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
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if n_sma > 4.0:
                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2).add_(group['eps'])
                    grad.div_(de_nom)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=False,
                    fixed_decay=False,
                )

                p.lerp_(z, weight=checkpoint)
                p.add_(grad, alpha=adaptive_y_lr)

                z.sub_(grad, alpha=lr)

        return loss


class ScheduleFreeWrapper(BaseOptimizer):
    r"""Wrap any optimizer to make it Schedule-Free.

        This version uses a memory-efficient swap operation but may be slower than the reference version. In most cases
        the performance difference is negligible. For the best possible performance and memory-usage, Schedule-Free
        needs to be directly integrated with the base optimizer.

        When using this version, you can disable the base optimizer's momentum, as it's no longer necessary when using
        our wrapper's momentum (although you can use both types of momentum if you want).

        If you set weight decay on the base optimizer, it computes weight decay at $z$. We offer the option to compute
        weight decay at $y$, via the `weight_decay_at_y` parameter, which seems to give better results in our
        experiments. This approach to decay only works correctly if the base optimizer uses group['lr'] as the current
        learning rate.

    :param optimizer: OPTIMIZER_INSTANCE_OR_CLASS. base optimizer.
    :param momentum: float. momentum.
    :param weight_decay: float. weight decay (L2 penalty).
    :param r: float. use polynomial weighting in the average with power r.
    :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
        set to 0 for no weighting.
    """

    def __init__(
        self,
        optimizer: OPTIMIZER_INSTANCE_OR_CLASS,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        **kwargs,
    ):
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.r = r
        self.weight_lr_power = weight_lr_power
        self.train_mode: bool = False

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.state: STATE = defaultdict(dict)
        self.defaults: DEFAULTS = self.optimizer.defaults

    def __str__(self) -> str:
        return 'ScheduleFree'

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return {'state': self.state, 'optimizer': self.optimizer}

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def state_dict(self) -> STATE:
        return {'schedulefree_state': self.state, 'base_optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state: STATE) -> None:
        r"""Load state."""
        self.state = state['schedulefree_state']
        self.optimizer.load_state_dict(state['base_optimizer'])

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    @torch.no_grad()
    def eval(self):
        if not self.train_mode:
            return

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'z' in state:
                    p.lerp_(end=state['z'], weight=1.0 - 1.0 / self.momentum)

        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if self.train_mode:
            return

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'z' in state:
                    p.lerp_(end=state['z'], weight=1.0 - self.momentum)

        self.train_mode = True

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor) -> None:
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if not self.train_mode:
            raise ValueError('optimizer was not in train mode when step is called. call .train() before training')

        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if 'z' not in state:
                    state['z'] = p.clone()

                z = state['z']

                self.apply_weight_decay(
                    z,
                    grad,
                    lr=group['lr'],
                    weight_decay=self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                )

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=group['lr'],
                    weight_decay=self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                    ratio=1.0 - self.momentum,
                )

                p.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)

                self.swap(z, p)

        self.optimizer.step()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr: float = group['lr'] * group.get('d', 1.0)
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))

            weight: float = (group['step'] ** group['lr']) * (lr_max ** self.weight_lr_power)  # fmt: skip
            weight_sum = group['weight_sum'] = group.get('weight_sum', 0.0) + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                z = state['z']

                self.swap(z, p)

                p.lerp_(end=z, weight=checkpoint)

                p.lerp_(end=state['z'], weight=1.0 - self.momentum)

        return loss
