from collections import defaultdict
from collections.abc import Callable

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State


class Lookahead(BaseOptimizer):
    """Wrap an optimizer with periodic interpolation toward slow weights.

    Args:
        optimizer: Base optimizer.
        k: Number of base optimizer steps between slow weight updates.
        alpha: Interpolation factor from slow weights toward fast weights.
        pullback_momentum: Momentum handling at interpolation: `'none'`, `'reset'`, or `'pullback'`.

    """

    def __init__(
        self,
        optimizer: OptimizerInstanceOrClass,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = 'none',
        **kwargs,
    ) -> None:
        self.validate_positive(k, 'k')
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_options(pullback_momentum, 'pullback_momentum', ['none', 'reset', 'pullback'])

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: dict[int, Callable] = {}
        self._optimizer_step_post_hooks: dict[int, Callable] = {}

        self.alpha = alpha
        self.k = k
        self.pullback_momentum = pullback_momentum

        self.state: State = defaultdict(dict)

        for group in self.param_groups:
            if 'counter' not in group:
                group['counter'] = 0

            for p in group['params']:
                state = self.state[p]
                state['slow_params'] = p.detach().clone()
                if self.pullback_momentum == 'pullback':
                    state['slow_momentum'] = torch.zeros_like(p)

        self.defaults: Defaults = {
            'lookahead_alpha': alpha,
            'lookahead_k': k,
            'lookahead_pullback_momentum': pullback_momentum,
            **self.optimizer.defaults,
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
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    def backup_and_load_cache(self) -> None:
        """Back up fast weights and load slow weights for evaluation."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['backup_params'] = p.detach().clone()
                p.data.copy_(state['slow_params'])

    def clear_and_load_backup(self) -> None:
        """Restore fast weights after evaluating slow weights."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['backup_params'])
                del state['backup_params']

    def state_dict(self) -> State:
        lookahead_state: State = {
            (group_index, parameter_index): dict(self.state[p])
            for group_index, group in enumerate(self.param_groups)
            for parameter_index, p in enumerate(group['params'])
            if p in self.state
        }
        return {'lookahead_state': lookahead_state, 'base_optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state: State) -> None:
        """Restore optimizer state and slow weights from a checkpoint."""
        saved_state = state['lookahead_state']
        restored_state: State = {}
        for group_index, group in enumerate(self.param_groups):
            for parameter_index, p in enumerate(group['params']):
                key = (group_index, parameter_index)
                if key in saved_state:
                    restored_state[p] = dict(saved_state[key])
                elif p in saved_state:
                    restored_state[p] = dict(saved_state[p])
        parameter_count = sum(len(group['params']) for group in self.param_groups)
        if len(restored_state) != len(saved_state) or len(restored_state) != parameter_count:
            raise ValueError('lookahead state does not match the current parameters')

        self.optimizer.load_state_dict(state['base_optimizer'])
        for p, parameter_state in restored_state.items():
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    parameter_state[key] = value.to(device=p.device, dtype=p.dtype).clone()
        self.state = defaultdict(dict, restored_state)

    @torch.no_grad()
    def update(self, group: dict):
        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]

            slow = state['slow_params']

            p.lerp_(slow, weight=1.0 - self.alpha)
            slow.copy_(p)

            if self.pullback_momentum == 'pullback':
                if 'momentum_buffer' not in self.optimizer.state[p]:
                    self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

                internal_momentum = self.optimizer.state[p]['momentum_buffer']
                internal_momentum.lerp_(state['slow_momentum'], weight=1.0 - self.alpha)
                state['slow_momentum'].copy_(internal_momentum)
            elif self.pullback_momentum == 'reset':
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
                self.update(group)
        return loss
