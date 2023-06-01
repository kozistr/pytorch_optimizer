import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

# Modified from https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py (MIT David Samuel)


class AdaHessian(Optimizer, BaseOptimizer):
    r"""An Adaptive Second Order Optimizer for Machine Learning

    Requires `loss.backward(create_graph=True)` in order to calculate hessians

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param hessian_power: float. exponent of the hessian trace
    :param update_period: int. number of steps after which to apply hessian approximation
    :param n_samples: int. times to sample `z` for the approximation of the hessian trace
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(self,
                 params: PARAMETERS,
                 lr: float = 1e-1,
                 betas: BETAS = (0.9, 0.999),
                 weight_decay: float = 0.0,
                 weight_decouple: bool = True,
                 fixed_decay: bool = False,
                 hessian_power: float = 1.0,
                 update_period: int = 1,
                 n_samples: int = 1,
                 eps: float = 1e-16):

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_range(hessian_power, "Hessian Power", 0, 1, range_type='(]')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'hessian_power': hessian_power,
            'update_period': update_period,
            'n_samples': n_samples,
            'eps': eps,
        }
        self._step = 0
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._step % self.update_period == 0:
            self.compute_hutchinson_hessian(self.n_samples)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                # State initialization
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                if self._step % self.update_period == 0:
                    exp_hessian_diag_sq.mul_(beta2).addcmul_(state['hessian'], state['hessian'], value=1 - beta2)

                bias_correction1 = 1 - beta1 ** self._step
                bias_correction2 = 1 - beta2 ** self._step

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        self._step += 1
        return loss
