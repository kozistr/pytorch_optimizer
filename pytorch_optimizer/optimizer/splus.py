from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import (
    BETAS,
    CLOSURE,
    DEFAULTS,
    GROUP,
    LOSS,
    PARAMETERS,
)


class SPlus(BaseOptimizer):
    r"""A Stable Whitening Optimizer for Efficient Neural Network Training.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ema_rate: float. exponential moving average decay rate.
    :param inverse_steps: int. the number of steps to perform inverse.
    :param nonstandard_constant: float. scale factor for learning rate in case of non-linear layer.
    :param max_dim: int. maximum number of dimensions to perform .
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-1,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ema_rate: float = 0.999,
        inverse_steps: int = 100,
        nonstandard_constant: float = 1e-3,
        max_dim: int = 10000,
        eps: float = 1e-30,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(ema_rate, 'ema_rate', 0.0, 1.0)
        self.validate_positive(inverse_steps, 'inverse_steps')
        self.validate_positive(max_dim, 'max_dim')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ema_rate': ema_rate,
            'inverse_steps': inverse_steps,
            'max_dim': max_dim,
            'nonstandard_constant': nonstandard_constant,
            'eps': eps,
            'train_mode': True,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SPlus'

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group.get('train_mode'):
                for p in group['params']:
                    state = self.state[p]
                    state['param_buffer'] = p.clone()
                    p.lerp_(state['ema'], weight=1.0).mul_(1.0 / (1.0 - group['ema_rate'] ** group['step']))
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if 'train_mode' in group and not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'param_buffer' in state:
                        p.lerp_(state['param_buffer'], weight=1.0)
                        del state['param_buffer']
                group['train_mode'] = True

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['momentum'] = torch.zeros_like(p)
                state['ema'] = torch.zeros_like(p)
                if len(p.shape) == 2:
                    state['sides'] = [
                        torch.zeros((d, d), device=p.device, dtype=p.dtype) if d < group['max_dim'] else None
                        for d in p.shape
                    ]
                    state['q_sides'] = [
                        torch.eye(d, device=p.device, dtype=p.dtype) if d < group['max_dim'] else None for d in p.shape
                    ]

    @staticmethod
    def get_scaled_lr(shape: Tuple[int, int], lr: float, nonstandard_constant: float, max_dim: int = 10000) -> float:
        scale: float = (
            nonstandard_constant
            if len(shape) != 2 or shape[0] > max_dim or shape[1] > max_dim
            else 2.0 / (shape[0] + shape[1])
        )
        return lr * scale

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                scaled_lr: float = self.get_scaled_lr(
                    p.shape, group['lr'], group['nonstandard_constant'], group['max_dim']
                )

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=scaled_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                m, ema = state['momentum'], state['ema']
                m.lerp_(grad, weight=1.0 - beta1)

                if len(p.shape) == 2:
                    sides, q_sides = state['sides'], state['q_sides']

                    m = q_sides[0].T @ m if q_sides[0] is not None else m
                    m = m @ q_sides[1] if q_sides[1] is not None else m

                    if sides[0] is not None:
                        torch.lerp(sides[0], grad @ grad.T, weight=1.0 - beta2, out=sides[0])

                    if sides[1] is not None:
                        torch.lerp(sides[1], grad.T @ grad, weight=1.0 - beta2, out=sides[1])

                    update = torch.sign(m)

                    if q_sides[0] is not None:
                        update = q_sides[0] @ update

                    if q_sides[1] is not None:
                        update = update @ q_sides[1].T

                    if group['step'] == 1 or group['step'] % group['inverse_steps'] == 0:
                        if sides[0] is not None:
                            _, eig_vecs = torch.linalg.eigh(
                                sides[0].float() + torch.eye(sides[0].shape[0], device=p.device).mul_(group['eps'])
                            )
                            state['q_sides'][0] = eig_vecs.to(sides[0].dtype)
                        if sides[1] is not None:
                            _, eig_vecs = torch.linalg.eigh(
                                sides[1].float() + torch.eye(sides[1].shape[0], device=p.device).mul_(group['eps'])
                            )
                            state['q_sides'][1] = eig_vecs.to(sides[1].dtype)
                else:
                    update = torch.sign(m)

                p.add_(update, alpha=-scaled_lr)

                ema.lerp_(p, weight=1.0 - group['ema_rate'])

        return loss
