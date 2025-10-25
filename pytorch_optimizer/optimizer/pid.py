import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


class PID(BaseOptimizer):
    r"""A PID Controller Approach for Stochastic Optimization of Deep Networks.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor.
    :param dampening: float. dampening for momentum.
    :param derivative: float. D part of the PID.
    :param integral: float. I part of the PID.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        derivative: float = 10.0,
        integral: float = 5.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(derivative, 'derivative')
        self.validate_non_negative(integral, 'integral')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'derivative': derivative,
            'integral': integral,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'PID'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0 and group['momentum'] > 0.0:
                state['grad_buffer'] = torch.zeros_like(p)
                state['i_buffer'] = torch.zeros_like(p)
                state['d_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                g_buf, i_buf, d_buf = (
                    state.get('grad_buffer', None),
                    state.get('i_buffer', None),
                    state.get('d_buffer', None),
                )

                p, grad, g_buf, i_buf, d_buf = self.view_as_real(p, grad, g_buf, i_buf, d_buf)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['momentum'] > 0.0:
                    i_buf.mul_(group['momentum']).add_(grad, alpha=1.0 - group['dampening'])
                    d_buf.mul_(group['momentum'])

                    if group['step'] > 1:
                        d_buf.add_(grad - g_buf, alpha=1.0 - group['momentum'])
                        g_buf.copy_(grad)

                    grad.add_(i_buf, alpha=group['integral']).add_(d_buf, alpha=group['derivative'])

                p.add_(grad, alpha=-group['lr'])

        return loss
