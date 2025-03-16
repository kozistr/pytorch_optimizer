import torch
from torch.nn.functional import normalize

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class TAM(BaseOptimizer):
    r"""Torque-Aware Momentum.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. coefficients used for computing running averages of gradient.
    :parma decay_rate: float. smoothing decay rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.9,
        decay_rate: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(decay_rate, 'decay_rate', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
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

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum: float = group['momentum']
            decay_rate: float = group['decay_rate']
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
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        decay_rate: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(decay_rate, 'decay_rate', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
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

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']
            decay_rate: float = group['decay_rate']

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
