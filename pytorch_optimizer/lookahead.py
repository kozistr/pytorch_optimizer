from collections import defaultdict
from typing import Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.types import (
    CLOSURE,
    DEFAULT_PARAMETERS,
    LOSS,
    PARAM_GROUP,
    PARAM_GROUPS,
    STATE,
)


class Lookahead(Optimizer):
    """
    Reference : https://github.com/alphadl/lookahead.pytorch/blob/master/lookahead.py
    Example :
        from pytorch_optimizer import AdamP, Lookahead
        ...
        model = YourModel()
        base_optimizer = AdamP(model.parameters())
        optimizer = Lookahead(base_optimizer)
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = 'none',
    ):
        """
        :param optimizer: Optimizer.
        :param k: int. number of lookahead steps
        :param alpha: float. linear interpolation factor
        :param pullback_momentum: str. change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum

        self.check_valid_parameters()

        self.param_groups: PARAM_GROUPS = self.optimizer.param_groups
        self.fast_state: STATE = self.optimizer.state
        self.state: STATE = defaultdict(dict)

        for group in self.param_groups:
            group['counter'] = 0

        self.defaults: DEFAULT_PARAMETERS = dict(
            k=k,
            alpha=alpha,
            pullback_momentum=pullback_momentum,
            **optimizer.defaults,
        )

    def check_valid_parameters(self):
        if 1 > self.k:
            raise ValueError(f'Invalid k : {self.k}')
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f'Invalid alpha : {self.alpha}')
        if self.pullback_momentum not in ('none', 'reset', 'pullback'):
            raise ValueError(
                f'Invalid pullback_momentum : {self.pullback_momentum}'
            )

    def update(self, group: Dict):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.zeros_like(fast.data)
                param_state['slow_param'].copy_(fast.data)
                if self.pullback_momentum == 'pullback':
                    param_state['slow_mom'] = torch.zeros_like(fast.data)

            slow = param_state['slow_param']
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

            if self.pullback_momentum == 'pullback':
                internal_momentum = self.optimizer.state[fast][
                    'momentum_buffer'
                ]
                self.optimizer.state[fast][
                    'momentum_buffer'
                ] = internal_momentum.mul_(self.alpha).add_(
                    1.0 - self.alpha, param_state['slow_mom']
                )
                param_state['slow_mom'] = self.optimizer.state[fast][
                    'momentum_buffer'
                ]
            elif self.pullback_momentum == 'reset':
                self.optimizer.state[fast][
                    'momentum_buffer'
                ] = torch.zeros_like(fast.data)

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
