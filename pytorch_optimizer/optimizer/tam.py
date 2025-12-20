import torch
from torch.nn.functional import normalize

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class TAM(BaseOptimizer):
    """Torque-Aware Momentum.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Coefficient used for computing running averages of gradient.
        decay_rate (float): Smoothing decay rate.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Whether to fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        momentum: float = 0.9,
        decay_rate: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(decay_rate, 'decay_rate', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'decay_rate': decay_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'TAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['s'] = torch.zeros_like(grad)
                state['momentum_buffer'] = grad.clone()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            momentum: float = group['momentum']
            decay_rate: float = group['decay_rate']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                s, momentum_buffer = state['s'], state['momentum_buffer']

                corr = normalize(momentum_buffer, p=2.0, dim=0).mul_(normalize(grad, p=2.0, dim=0))
                s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

                d = ((1.0 + s) / 2.0).add_(group['eps']).mul_(grad)

                momentum_buffer.mul_(momentum).add_(d)

                self.apply_weight_decay(
                    p,
                    grad,
                    group['lr'],
                    group['weight_decay'],
                    group['weight_decouple'],
                    group['fixed_decay'],
                )

                p.add_(momentum_buffer, alpha=-group['lr'])

        return loss


class AdaTAM(BaseOptimizer):
    r"""Adaptive Torque-Aware Momentum.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :parma decay_rate: float. smoothing decay rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        decay_rate: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(decay_rate, 'decay_rate', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'decay_rate': decay_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaTAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['s'] = torch.zeros_like(grad)
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']
            decay_rate: float = group['decay_rate']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad,
                    group['lr'],
                    group['weight_decay'],
                    group['weight_decouple'],
                    group['fixed_decay'],
                )

                s, exp_avg, exp_avg_sq = state['s'], state['exp_avg'], state['exp_avg_sq']

                corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(grad, p=2.0, dim=0))
                s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

                d = ((1.0 + s) / 2.0).add_(group['eps']).mul_(grad)

                exp_avg.mul_(beta1).add_(d)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-group['lr'])

        return loss
