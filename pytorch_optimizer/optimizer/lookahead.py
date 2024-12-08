from collections import defaultdict
from typing import Callable, Dict

import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, OPTIMIZER, STATE


class Lookahead(BaseOptimizer):
    r"""k steps forward, 1 step back.

    :param optimizer: OPTIMIZER. base optimizer.
    :param k: int. number of lookahead steps.
    :param alpha: float. linear interpolation factor.
    :param pullback_momentum: str. change to inner optimizer momentum on interpolation update.
    """

    def __init__(
        self,
        optimizer: OPTIMIZER,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = 'none',
        **kwargs,
    ) -> None:
        self.validate_positive(k, 'k')
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_options(pullback_momentum, 'pullback_momentum', ['none', 'reset', 'pullback'])

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.pullback_momentum = pullback_momentum

        self.state: STATE = defaultdict(dict)

        for group in self.param_groups:
            if 'counter' not in group:
                group['counter'] = 0

            for p in group['params']:
                state = self.state[p]
                state['slow_params'] = torch.empty_like(p)
                state['slow_params'].copy_(p)
                if self.pullback_momentum == 'pullback':
                    state['slow_momentum'] = torch.zeros_like(p)

        self.defaults: DEFAULTS = {
            'lookahead_alpha': alpha,
            'lookahead_k': k,
            'lookahead_pullback_momentum': pullback_momentum,
            **optimizer.defaults,
        }

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum,
        }

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['counter'] = 0

    def backup_and_load_cache(self):
        r"""Backup cache parameters."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['backup_params'] = torch.empty_like(p)
                state['backup_params'].copy_(p)
                p.data.copy_(state['slow_params'])

    def clear_and_load_backup(self):
        r"""Load backup parameters."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['backup_params'])
                del state['backup_params']

    def state_dict(self) -> STATE:
        return {'lookahead_state': self.state, 'base_optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state: STATE):
        r"""Load state."""
        self.state = state['lookahead_state']
        self.optimizer.load_state_dict(state['base_optimizer'])

    @torch.no_grad()
    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def update(self, group: Dict):
        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]

            slow = state['slow_params']

            p.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.copy_(p)

            if 'momentum_buffer' not in self.optimizer.state[p]:
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

            if self.pullback_momentum == 'pullback':
                internal_momentum = self.optimizer.state[p]['momentum_buffer']
                self.optimizer.state[p]['momentum_buffer'] = internal_momentum.mul_(self.alpha).add_(
                    state['slow_momentum'], alpha=1.0 - self.alpha
                )
                state['slow_momentum'] = self.optimizer.state[p]['momentum_buffer']
            elif self.pullback_momentum == 'reset':
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = self.optimizer.step(closure)
        for group in self.param_groups:
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
                self.update(group)
        return loss
