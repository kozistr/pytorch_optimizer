from collections import defaultdict
from typing import Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, OPTIMIZER, STATE


class Lookahead(Optimizer, BaseOptimizer):
    r"""k steps forward, 1 step back

    :param optimizer: OPTIMIZER. base optimizer.
    :param k: int. number of lookahead steps.
    :param alpha: float. linear interpolation factor.
    :param pullback_momentum: str. change to inner optimizer momentum on interpolation update.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        optimizer: OPTIMIZER,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = 'none',
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum

        self.validate_parameters()

        self.param_groups = self.optimizer.param_groups
        self.fast_state: STATE = self.optimizer.state
        self.state: STATE = defaultdict(dict)
        self.reset()

        self.defaults: DEFAULTS = optimizer.defaults
        self.defaults.update(
            {
                'k': k,
                'alpha': alpha,
                'pullback_momentum': pullback_momentum,
            }
        )

    def validate_parameters(self):
        self.validate_lookahead_k(self.k)
        self.validate_alpha(self.alpha)
        self.validate_pullback_momentum(self.pullback_momentum)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['counter'] = 0

    @torch.no_grad()
    def update(self, group: Dict):
        for fast in group['params']:
            if fast.grad is None:
                continue

            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.empty_like(fast)
                param_state['slow_param'].copy_(fast)
                if self.pullback_momentum == 'pullback':
                    param_state['slow_mom'] = torch.zeros_like(fast)

            slow = param_state['slow_param']
            slow += (fast - slow) * self.alpha
            fast.copy_(slow)

            if 'momentum_buffer' not in self.optimizer.state[fast]:
                self.optimizer.state[fast]['momentum_buffer'] = torch.zeros_like(fast)

            if self.pullback_momentum == 'pullback':
                internal_momentum = self.optimizer.state[fast]['momentum_buffer']
                self.optimizer.state[fast]['momentum_buffer'] = internal_momentum.mul_(self.alpha).add_(
                    param_state['slow_mom'], alpha=1.0 - self.alpha
                )
                param_state['slow_mom'] = self.optimizer.state[fast]['momentum_buffer']
            elif self.pullback_momentum == 'reset':
                self.optimizer.state[fast]['momentum_buffer'] = torch.zeros_like(fast)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = self.optimizer.step(closure)
        for group in self.param_groups:
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
                self.update(group)
        return loss

    def state_dict(self) -> STATE:
        fast_state_dict: STATE = self.optimizer.state_dict()
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']

        slow_state: STATE = {(id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}

        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict: STATE):
        slow_state_dict: STATE = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict: STATE = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super().load_state_dict(slow_state_dict)

        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group['counter'] = 0
        self.optimizer.add_param_group(param_group)
