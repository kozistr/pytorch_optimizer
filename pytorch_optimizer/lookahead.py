from collections import defaultdict
from typing import Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.types import (
    CLOSURE,
    LOSS,
    PARAM_GROUP,
    PARAM_GROUPS,
    STATE,
)


class Lookahead(Optimizer):
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        self.param_groups: PARAM_GROUPS = self.optimizer.param_groups
        self.fast_state: STATE = self.optimizer.state
        self.state: STATE = defaultdict(dict)

        for group in self.param_groups:
            group['counter'] = 0

    def update(self, group: Dict):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.zeros_like(fast.data)
                param_state['slow_param'].copy_(fast.data)
            slow = param_state['slow_param']
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = self.optimizer.step(closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self.update(group)
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
        return loss

    def state_dict(self) -> STATE:
        fast_state_dict: STATE = self.optimizer.state_dict()
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']

        slow_state: STATE = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }

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

    def add_param_group(self, param_group: PARAM_GROUP):
        param_group['counter'] = 0
        self.optimizer.add_param_group(param_group)
