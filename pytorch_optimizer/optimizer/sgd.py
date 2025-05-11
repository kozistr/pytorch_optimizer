import math
from typing import Dict, Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


class AccSGD(BaseOptimizer):
    r"""Accelerating Stochastic Gradient Descent For Least Squares Regression.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param kappa: float. ratio of long to short step.
    :param xi: float. statistical advantage parameter.
    :param constant: float. any small constant under 1.
    :param weight_decay: float. weight decay.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        kappa: float = 1000.0,
        xi: float = 10.0,
        constant: float = 0.7,
        weight_decay: float = 0.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(kappa, 'kappa')
        self.validate_non_negative(xi, 'xi')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_boundary(constant, boundary=1.0, bound_type='upper')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'kappa': kappa,
            'xi': xi,
            'constant': constant,
            'weight_decay': weight_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AccSGD'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = p.clone()

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

            large_lr: float = group['lr'] * group['kappa'] / group['constant']
            alpha: float = 1.0 - (group['xi'] * (group['constant'] ** 2) / group['kappa'])
            beta: float = 1.0 - alpha
            zeta: float = group['constant'] / (group['constant'] + beta)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=False,
                    fixed_decay=False,
                )

                buf = state['momentum_buffer']
                buf.mul_((1.0 / beta) - 1.0).add_(grad, alpha=-large_lr).add_(p).mul_(beta)

                p.add_(grad, alpha=-group['lr']).mul_(zeta).add_(buf, alpha=1.0 - zeta)

        return loss


class SGDW(BaseOptimizer):
    r"""Decoupled Weight Decay Regularization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param dampening: float. dampening for momentum.
    :param nesterov: bool. enables Nesterov momentum.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-4,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        dampening: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'dampening': dampening,
            'nesterov': nesterov,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDW'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = p.clone()

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

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if momentum > 0.0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                    if group['nesterov']:
                        grad.add_(buf, alpha=momentum)
                    else:
                        grad = buf

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                p.add_(grad, alpha=-group['lr'])

        return loss


class ASGD(BaseOptimizer):
    r"""Adaptive SGD with estimation of the local smoothness (curvature).

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param amplifier: float. amplifier.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param theta: float. theta.
    :param dampening: float. dampening for momentum.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        amplifier: float = 0.02,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        theta: float = 1.0,
        dampening: float = 1.0,
        eps: float = 1e-5,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(amplifier, 'amplifier')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'amplifier': amplifier,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'theta': theta,
            'dampening': dampening,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ASGD'

    def init_group(self, group: GROUP, **kwargs) -> None:
        pass

    @staticmethod
    def get_norms_by_group(group: GROUP, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Get parameter & gradient norm by group."""
        p_norm = torch.zeros(1, dtype=torch.float32, device=device)
        g_norm = torch.zeros(1, dtype=torch.float32, device=device)

        for p in group['params']:
            if p.grad is None:
                continue

            p_norm.add_(p.norm().pow(2))
            g_norm.add_(p.grad.norm().pow(2))

        p_norm.sqrt_()
        g_norm.sqrt_()

        return p_norm, g_norm

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            device = group['params'][0].device

            if 'prev_param_norm' not in group and 'prev_grad_norm' not in group:
                group['prev_param_norm'], group['prev_grad_norm'] = self.get_norms_by_group(group, device)

            group['curr_param_norm'], group['curr_grad_norm'] = self.get_norms_by_group(group, device)

            param_diff_norm: float = (group['curr_param_norm'] - group['prev_param_norm']).item()
            grad_diff_norm: float = (group['curr_grad_norm'] - group['prev_grad_norm']).item()

            new_lr: float = group['lr'] * math.sqrt(1 + group['amplifier'] * group['theta'])
            if param_diff_norm > 0 and grad_diff_norm > 0:
                new_lr = min(new_lr, param_diff_norm / (group['dampening'] * grad_diff_norm)) + group['eps']

            group['theta'] = new_lr / group['lr']
            group['lr'] = new_lr

            group['prev_param_norm'].copy_(group['curr_param_norm'])
            group['prev_grad_norm'].copy_(group['curr_grad_norm'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.add_(grad, alpha=-new_lr)

        return loss


class SignSGD(BaseOptimizer):
    r"""Compressed Optimisation for Non-Convex Problems.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor (0.0 = SignSGD, >0 = Signum).
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SignSGD'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if group['momentum'] > 0.0:
                state['momentum_buffer'] = torch.zeros_like(p)

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

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if momentum > 0.0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                else:
                    buf = grad

                p.add_(torch.sign(buf) if not torch.is_complex(buf) else torch.sgn(buf), alpha=-group['lr'])

        return loss


class SGDSaI(BaseOptimizer):
    r"""No More Adam: Learning Rate Scaling at Initialization is All You Need.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. coefficients used for computing running averages of gradient.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.has_warmup: bool = False
        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDSaI'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if group['momentum'] > 0.0:
                state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def warmup_step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                sigma = grad.std().nan_to_num_() if grad.ndim > 1 and grad.size(0) != 1 else 0
                grad_norm = grad.norm()

                g_snr = grad_norm.div_(sigma.add_(group['eps'])) if sigma != 0.0 else grad_norm

                self.state[p]['gsnr'] = g_snr

        self.has_warmup = True

        return loss

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if not self.has_warmup:
            self.warmup_step(closure)

        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['step'] += 1

            momentum: float = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if momentum > 0.0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                else:
                    buf = grad

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                p.add_(buf, alpha=-group['lr'] * state['gsnr'])

        return loss


class VSGD(BaseOptimizer):
    r"""Variational Stochastic Gradient Descent for Deep Neural Networks.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param ghattg: float. prior variance ratio between ghat and g, Var(ghat_t-g_t)/Var(g_t-g_{t-1}).
    :param ps: float. prior strength.
    :param tau1: float. remember rate for the gamma parameters of g.
    :param tau2: float. remember rate for the gamma parameter of ghat.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-1,
        ghattg: float = 30.0,
        ps: float = 1e-8,
        tau1: float = 0.81,
        tau2: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(ghattg, 'ghattg')
        self.validate_non_negative(ps, 'ps')
        self.validate_non_negative(tau1, 'tau1')
        self.validate_non_negative(tau2, 'tau2')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'tau1': tau1,
            'tau2': tau2,
            'pa2': 2.0 * ps + 1.0 + 1e-4,
            'pbg2': 2.0 * ps,
            'pbhg2': 2.0 * ghattg * ps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'VSGD'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['mug'] = torch.zeros_like(p)
                state['bg'] = torch.zeros_like(p)
                state['bhg'] = torch.zeros_like(p)

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

            pa2, pbg2, pbhg2 = group['pa2'], group['pbg2'], group['pbhg2']

            rho1: float = math.pow(group['step'], -group['tau1'])
            rho2: float = math.pow(group['step'], -group['tau2'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                bg, bhg = state['bg'], state['bhg']

                if group['step'] == 1:
                    sg = pbg2 / (pa2 - 1.0)
                    shg = pbhg2 / (pa2 - 1.0)
                else:
                    sg = bg / pa2
                    shg = bhg / pa2

                mug = state['mug']
                mug_prev = mug.clone()

                mug.mul_(shg).add_(grad * sg).div_(sg + shg)

                sigg = (sg * shg) / (sg + shg)
                mug_sq = mug.pow(2).add_(sigg)

                bg2 = pbg2 + mug_sq - 2.0 * mug * mug_prev + mug_prev.pow(2)
                bhg2 = pbhg2 + mug_sq - 2.0 * grad * mug + grad.pow(2)

                bg.mul_(1.0 - rho1).add_(bg2, alpha=rho1)
                bhg.mul_(1.0 - rho2).add_(bhg2, alpha=rho2)

                p.add_(group['lr'] / mug_sq.sqrt().add_(group['eps']) * mug, alpha=-1.0)

        return loss
