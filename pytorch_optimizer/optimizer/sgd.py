import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class AccSGD(BaseOptimizer):
    """Accelerated SGD with coupled short and long steps.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        kappa: Ratio of long to short step.
        xi: Statistical advantage parameter.
        constant: Any small constant under 1.
        weight_decay: Weight decay coefficient.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
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

        defaults: Defaults = {
            'lr': lr,
            'kappa': kappa,
            'xi': xi,
            'constant': constant,
            'weight_decay': weight_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AccSGD'

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
                state['momentum_buffer'] = p.clone()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
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

                p.add_(grad, alpha=-group['lr']).lerp_(buf, weight=1.0 - zeta)

        return loss


class SGDW(BaseOptimizer):
    """SGD with optional decoupled weight decay.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        dampening: Dampening factor for momentum.
        nesterov: Use Nesterov momentum.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        dampening: float = 0.0,
        nesterov: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'dampening': dampening,
            'nesterov': nesterov,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDW'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> None:
        lr, momentum, dampening = group['lr'], group['momentum'], group['dampening']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=False,
        )

        if momentum > 0.0:
            buffers, existing_buffers, existing_grads = [], [], []
            for p, grad in zip(params, grads):
                state = self.state[p]
                buf = state.get('momentum_buffer')
                if buf is None:
                    state['momentum_buffer'] = buf = grad.clone()
                else:
                    existing_buffers.append(buf)
                    existing_grads.append(grad)
                buffers.append(buf)

            if existing_buffers:
                torch._foreach_mul_(existing_buffers, momentum)
                torch._foreach_add_(existing_buffers, existing_grads, alpha=1.0 - dampening)

            grads = torch._foreach_add(grads, buffers, alpha=momentum) if group['nesterov'] else buffers

        torch._foreach_add_(params, grads, alpha=-lr)

    def _step_per_param(self, group: ParamGroup) -> None:
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            self.apply_weight_decay(
                p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

            if momentum > 0.0:
                state = self.state[p]
                buf = state.get('momentum_buffer')
                if buf is None:
                    state['momentum_buffer'] = buf = grad.clone()
                else:
                    buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                grad = grad.add_(buf, alpha=momentum) if group['nesterov'] else buf

            p.add_(grad, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                if params:
                    self._step_foreach(group, params, grads)
            else:
                self._step_per_param(group)

        return loss


class ASGD(BaseOptimizer):
    """Adaptive SGD with estimation of the local smoothness (curvature).

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        amplifier: Coefficient controlling the maximum learning rate growth per step.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        theta: Initial ratio of consecutive learning rates, updated after each step.
        dampening: Scale of the local smoothness bound on the learning rate.
        eps: Term added to denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
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

        defaults: Defaults = {
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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_norms_by_group(group: ParamGroup, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute global parameter and gradient L2 norms for a parameter group."""
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
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
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

                self.maximize_gradient(grad, maximize=self.maximize)

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
    """Sign based SGD with optional momentum.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor. `0` gives SignSGD. Positive values give Signum.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SignSGD'

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

            if group['momentum'] > 0.0 and 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False or group['momentum'] == 0.0:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        momentum_buffers: list[torch.Tensor],
    ) -> None:
        lr = group['lr']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=lr,
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=False,
        )

        torch._foreach_lerp_(momentum_buffers, grads, weight=1.0 - group['momentum'])

        updates = [buf.sign() for buf in momentum_buffers]
        torch._foreach_add_(params, updates, alpha=-lr)

    def _step_per_param(self, group: ParamGroup) -> None:
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            self.apply_weight_decay(
                p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

            state = self.state[p]

            if momentum > 0.0:
                buf = state['momentum_buffer']
                buf.lerp_(grad, weight=1.0 - momentum)
            else:
                buf = grad

            p.add_(torch.sign(buf) if not torch.is_complex(buf) else torch.sgn(buf), alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['momentum_buffer']
                )
                if params:
                    self._step_foreach(group, params, grads, state_dict['momentum_buffer'])
            else:
                self._step_per_param(group)

        return loss


class SGDSaI(BaseOptimizer):
    """SGD with learning rate scaling from the initial gradient signal-to-noise ratio.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        eps: Term added to denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
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

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDSaI'

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

            if group['momentum'] > 0.0 and 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)

            if 'gsnr' not in state:
                sigma = grad.std().nan_to_num_() if grad.ndim > 1 and grad.size(0) != 1 else 0
                grad_norm = grad.norm()
                state['gsnr'] = grad_norm / (sigma + group['eps']) if sigma != 0.0 else grad_norm

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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if momentum > 0.0:
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, weight=1.0 - momentum)
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
    """SGD with variational gradient estimates.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        ghattg: Prior variance ratio between ghat and g, Var(ghat_t-g_t)/Var(g_t-g_{t-1}).
        ps: Prior strength.
        tau1: Remember rate for the gamma parameters of g.
        tau2: Remember rate for the gamma parameter of ghat.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        eps: Term added to denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
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

        defaults: Defaults = {
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
                state['mug'] = torch.zeros_like(p)
                state['bg'] = torch.full_like(p, group['pbg2'])
                state['bhg'] = torch.full_like(p, group['pbhg2'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
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

                bg.lerp_(bg2, weight=rho1)
                bhg.lerp_(bhg2, weight=rho2)

                p.add_(group['lr'] / mug_sq.sqrt().add_(group['eps']) * mug, alpha=-1.0)

        return loss
