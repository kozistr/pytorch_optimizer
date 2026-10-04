from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, cast

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce, get_world_size, is_initialized
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, OptimizerType, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import disable_running_stats, enable_running_stats


def get_global_gradient_norm(param_groups: list[ParamGroup], device: torch.device) -> torch.Tensor:
    """Compute the global L2 gradient norm for SAM perturbations.

    Args:
        param_groups: Optimizer groups. Adaptive groups weight gradients by the absolute parameter values.
        device: Device for the returned norm.

    Returns:
        torch.Tensor: Scalar gradient norm, or zero if no gradients are present.

    """
    norms: list[torch.Tensor] = []
    for group in param_groups or []:
        params: list[torch.Tensor] = group.get('params', []) or []
        adaptive: bool = group.get('adaptive', False)
        for p in params:
            if p.grad is not None:
                norm = ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(device)
                norms.append(norm)

    if not norms:
        return torch.tensor(0.0, device=device)

    return torch.norm(torch.stack(norms), p=2)


class SAM(BaseOptimizer):
    """Sharpness-aware minimization with a two pass parameter update.

    Compute gradients at the current weights before calling `step()`. The closure
    must recompute the loss and gradients at the perturbed weights.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        base_optimizer: Optimizer class to instantiate for the parameter update.
        rho: Radius of the neighborhood used to perturb parameters.
        use_gc: Centralize gradients before perturbing parameters.
        adaptive: Scale perturbations by the squared parameter values.
        perturb_eps: Stability constant for the perturbation norm.
        **kwargs (dict): Options for the base optimizer.

    Examples:
        ```python
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-3)
        for inputs, targets in data:
            optimizer.zero_grad()

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model(inputs), targets)
                loss.backward()
                return loss

            closure()
            optimizer.step(closure)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive, **kwargs}

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def __str__(self) -> str:
        return 'SAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                self.state[p]['old_p'] = p.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if 'old_p' in self.state[p]:
                    p.copy_(self.state[p].pop('old_p'))

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Closure = None):
        """Perturb weights, recompute gradients, and apply the base optimizer update.

        Args:
            closure: Callable that clears gradients and recomputes the loss and gradients. Compute the initial
                gradients before calling this method.

        Raises:
            NoClosureError: No closure is supplied.

        """
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.state = self.state  # ty: ignore[invalid-assignment]


class GSAM(BaseOptimizer):  # pragma: no cover
    """Sharpness-aware minimization with surrogate gap gradient decomposition.

    Use `set_closure()` to supply the loss and batch before each step. Advance the learning
    rate scheduler and call `update_rho_t()` to update the perturbation radius.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        base_optimizer: Existing optimizer instance for parameter updates.
        model: Model used for the forward passes.
        rho_scheduler (ProportionScheduler): Scheduler that supplies the perturbation radius.
        alpha: Weight of the surrogate gap gradient component.
        adaptive: Scale perturbations by the squared parameter values.
        perturb_eps: Stability constant for the perturbation norm.
        **kwargs (dict): Additional parameter group options.

    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: Optimizer,
        model: nn.Module,
        rho_scheduler,
        alpha: float = 0.4,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_range(alpha, 'alpha', 0.0, 1.0)

        self.model = model
        self.rho_scheduler = rho_scheduler
        self.alpha = alpha
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps

        self.rho_t: float = 0.0
        self.forward_backward_func: Callable | None = None

        if hasattr(ReduceOp, 'AVG'):
            self.grad_reduce = ReduceOp.AVG
            self.manual_average: bool = False
        else:
            self.grad_reduce = ReduceOp.SUM
            self.manual_average: bool = True

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        defaults: Defaults = {'adaptive': adaptive, **kwargs}

        super().__init__(params, defaults)

        self.update_rho_t()

    def __str__(self) -> str:
        return 'GSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def update_rho_t(self) -> float:
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho: float):
        grad_norm = self.grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['old_g'] = p.grad.clone()

                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def un_perturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha: float = 0.0):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                inner_prod += torch.sum(self.state[p]['old_g'] * p.grad)

        new_grad_norm = self.grad_norm(by=None)
        old_grad_norm = self.grad_norm(by='old_g')

        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def sync_grad(self):
        if is_initialized():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    all_reduce(p.grad, op=self.grad_reduce)
                    if self.manual_average:
                        p.grad.div_(float(get_world_size()))

    @torch.no_grad()
    def grad_norm(self, by: str | None = None, weight_adaptive: bool = False) -> torch.Tensor:
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if weight_adaptive else 1.0) * (p.grad if not by else self.state[p][by])).norm(p=2)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def maybe_no_sync(self):
        if is_initialized() and hasattr(self.model, 'no_sync'):
            return self.model.no_sync()  # ty: ignore[call-non-callable]
        return ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Store a forward backward closure for the current batch.

        The closure clears gradients, evaluates the model and loss, and runs backpropagation.

        Args:
            loss_fn: Callable accepting model predictions and targets.
            inputs: Model inputs for the current batch.
            targets: Target values for the current batch.
            **kwargs (dict): Additional arguments for the loss function.

        """

        def get_grad() -> tuple[Any, torch.Tensor]:
            self.base_optimizer.zero_grad()

            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)

            loss.backward()

            return outputs, loss.detach()

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure: Closure = None) -> tuple[Any, torch.Tensor]:
        get_grad = cast(Callable[[], tuple[Any, torch.Tensor]], closure or self.forward_backward_func)

        with self.maybe_no_sync():
            outputs, loss = get_grad()

            self.perturb_weights(rho=self.rho_t)

            disable_running_stats(self.model)

            get_grad()

            self.gradient_decompose(self.alpha)

            self.un_perturb()

        self.sync_grad()

        self.base_optimizer.step()

        enable_running_stats(self.model)

        return outputs, loss

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['base_optimizer'] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if 'base_optimizer' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
            self.param_groups = self.base_optimizer.param_groups
        else:
            self.base_optimizer.param_groups = self.param_groups


class WSAM(BaseOptimizer):
    """Sharpness-aware minimization with weighted sharpness regularization.

    Args:
        model: Model used for training. Supports DistributedDataParallel.
        params: Parameters to optimize or dictionaries defining parameter groups.
        base_optimizer: Optimizer class to instantiate for parameter updates.
        rho: Size of the neighborhood for computing the max loss.
        gamma: Sharpness mixing coefficient, used as `gamma / (1 - gamma)`.
        adaptive: Elementwise adaptive SAM.
        decouple: Apply the sharpness correction after the base optimizer update.
        max_norm: Max norm of the gradients.
        eps: Term added to the denominator of WSAM to improve numerical stability.
        **kwargs (dict): Parameters for optimizer.

    """

    def __init__(
        self,
        model: nn.Module | DistributedDataParallel,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        gamma: float = 0.9,
        adaptive: bool = False,
        decouple: bool = True,
        max_norm: float | None = None,
        eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')

        self.model = model
        self.decouple = decouple
        self.max_norm = max_norm

        alpha: float = gamma / (1.0 - gamma)

        defaults: Defaults = {'rho': rho, 'alpha': alpha, 'adaptive': adaptive, 'sam_eps': eps, **kwargs}

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'WSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device)

        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + group['sam_eps'])

            for p in group['params']:
                if p.grad is None:
                    continue

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]['e_w'] = e_w

                if is_initialized():  # pragma: no cover
                    all_reduce(p.grad, op=ReduceOp.AVG)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p].pop('grad', None)
                if p.grad is None:
                    continue

                self.state[p]['grad'] = p.grad.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.sub_(self.state[p].pop('e_w'))
                if p.grad is None:
                    continue

                if is_initialized():  # pragma: no cover
                    all_reduce(p.grad, ReduceOp.AVG)

        if self.max_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_norm)

        for group in self.param_groups:
            for p in group['params']:
                old_grad = self.state[p].pop('grad', None)
                if p.grad is None:
                    continue

                if old_grad is None:
                    old_grad = torch.zeros_like(p.grad)
                if not self.decouple:
                    p.grad.lerp_(old_grad, weight=1.0 - group['alpha'])
                else:
                    self.state[p]['sharpness'] = p.grad.clone() - old_grad
                    p.grad.copy_(old_grad)

        self.base_optimizer.step()

        if self.decouple:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    p.add_(self.state[p]['sharpness'], alpha=-group['lr'] * group['alpha'])

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Closure = None):
        if closure is None:
            raise NoClosureError(str(self))

        closure = torch.enable_grad()(closure)

        enable_running_stats(self.model)
        loss = closure()

        self.first_step(zero_grad=True)

        disable_running_stats(self.model)
        closure()

        self.second_step()

        return loss

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['base_optimizer'] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if 'base_optimizer' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
            self.param_groups = self.base_optimizer.param_groups
        else:
            self.base_optimizer.param_groups = self.param_groups


class BSAM(BaseOptimizer):
    """Bayesian sharpness-aware minimization with noisy parameter perturbations.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        num_data: Number of training data.
        lr: Learning rate.
        betas: Decay rates for gradient momentum and the squared curvature estimate.
        weight_decay: Weight decay coefficient.
        rho: Size of the neighborhood for computing the max loss.
        adaptive: Elementwise Adaptive SAM.
        damping: Damping to stabilize the method.
        **kwargs (dict): Parameters for optimizer.

    """

    def __init__(
        self,
        params: ParamsT,
        num_data: int,
        lr: float = 5e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-4,
        rho: float = 0.05,
        adaptive: bool = False,
        damping: float = 0.1,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(num_data, 'num_data')
        self.validate_non_negative(damping, 'damping')

        self.num_data = num_data
        self.damping = damping

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'rho': rho,
            'adaptive': adaptive,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'bSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]

            if 's' not in state:
                state['s'] = torch.ones_like(p)
                state['noisy_gradient'] = torch.zeros_like(p.grad)
                state['momentum'] = torch.zeros_like(p)

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                noise = torch.normal(0.0, 1 / (self.num_data * state['s']))

                p.add_(noise)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['noisy_gradient'] = p.grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * group['rho'] * p.grad / state['s']

                p.add_(e_w)

    @torch.no_grad()
    def third_step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                momentum, s = state['momentum'], state['s']
                momentum.lerp_(p.grad * weight_decay, weight=1.0 - beta1)

                var = (torch.sqrt(s).mul_(p.grad.abs()).add_(weight_decay + self.damping)).pow_(2)
                s.lerp_(var, weight=1.0 - beta2)

                p.add_(momentum / s, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None):
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step()

        with torch.enable_grad():
            closure()

        self.second_step()

        with torch.enable_grad():
            loss = closure()

        self.third_step()

        return loss


class LookSAM(BaseOptimizer):
    """Sharpness-aware minimization with periodic perturbation updates.

    Compute gradients at the current weights before calling `step()`. The closure
    must recompute the loss and gradients at the perturbed weights.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        base_optimizer: Optimizer class to instantiate for the parameter update.
        rho: Radius of the neighborhood used to perturb parameters.
        k: Number of steps between full sharpness gradient updates.
        alpha: Weight of the reused orthogonal sharpness gradient.
        use_gc: Centralize gradients before perturbing parameters.
        adaptive: Scale perturbations by the squared parameter values.
        perturb_eps: Stability constant for the perturbation norm.
        **kwargs (dict): Options for the base optimizer.

    Examples:
        ```python
        optimizer = LookSAM(model.parameters(), torch.optim.AdamW, lr=1e-3)
        for inputs, targets in data:
            optimizer.zero_grad()

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model(inputs), targets)
                loss.backward()
                return loss

            closure()
            optimizer.step(closure)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.1,
        k: int = 10,
        alpha: float = 0.7,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_positive(k, 'k')
        self.validate_range(alpha, 'alpha', 0.0, 1.0, '()')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.k = k
        self.alpha = alpha
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'LookSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def get_step(self):
        return (
            self.param_groups[0]['step']
            if 'step' in self.param_groups[0]
            else next(iter(self.base_optimizer.state.values()))['step'] if self.base_optimizer.state else 0
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        if self.get_step() % self.k != 0:
            return

        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for p in group['params']:
                self.state[p].pop('old_grad_p', None)
                if p.grad is None:
                    continue

                grad = p.grad
                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                self.state[p]['old_p'] = p.clone()
                self.state[p]['old_grad_p'] = grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        step = self.get_step()

        for group in self.param_groups:
            for p in group['params']:
                if 'old_p' in self.state[p]:
                    p.copy_(self.state[p].pop('old_p'))
                old_grad_p = self.state[p].pop('old_grad_p', None)
                if p.grad is None:
                    continue

                grad = p.grad
                grad_norm = grad.norm(p=2)

                if step % self.k == 0 and old_grad_p is not None:
                    g_grad_norm = old_grad_p / old_grad_p.norm(p=2).clamp_min(self.perturb_eps)
                    g_s_grad_norm = grad / grad_norm.clamp_min(self.perturb_eps)

                    self.state[p]['gv'] = torch.sub(
                        grad, grad_norm * torch.sum(g_grad_norm * g_s_grad_norm) * g_grad_norm
                    )
                elif step % self.k != 0 and 'gv' in self.state[p]:
                    gv = self.state[p]['gv']
                    grad.add_(grad_norm / (gv.norm(p=2) + 1e-8) * gv, alpha=self.alpha)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Closure = None):
        """Perturb weights, recompute gradients, and apply the base optimizer update.

        Args:
            closure: Callable that clears gradients and recomputes the loss and gradients. Compute the initial
                gradients before calling this method.

        Raises:
            NoClosureError: No closure is supplied.

        """
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['base_optimizer'] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if 'base_optimizer' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
            self.param_groups = self.base_optimizer.param_groups
        else:
            self.base_optimizer.param_groups = self.param_groups


class FriendlySAM(BaseOptimizer):
    """Sharpness-aware minimization with momentum adjusted perturbations.

    Compute gradients at the current weights before calling `step()`. The closure
    must recompute the loss and gradients at the perturbed weights.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        base_optimizer: Optimizer class to instantiate for the parameter update.
        rho: Radius of the neighborhood used to perturb parameters.
        sigma: Strength of the momentum subtraction in the perturbation gradient.
        lmbda: Decay rate for perturbation gradient momentum.
        adaptive: Scale perturbations by the squared parameter values.
        perturb_eps: Stability constant for the perturbation norm.
        **kwargs (dict): Options for the base optimizer.

    Examples:
        ```python
        optimizer = FriendlySAM(model.parameters(), torch.optim.AdamW, lr=1e-3)
        for inputs, targets in data:
            optimizer.zero_grad()

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model(inputs), targets)
                loss.backward()
                return loss

            closure()
            optimizer.step(closure)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        sigma: float = 1.0,
        lmbda: float = 0.9,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(sigma, 'sigma')
        self.validate_non_negative(lmbda, 'lmbda')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'sigma': sigma, 'lmbda': lmbda, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'FriendlySAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'momentum' not in state:
                    state['momentum'] = grad.clone()
                else:
                    momentum = state['momentum']

                    grad.sub_(momentum, alpha=group['sigma'])
                    momentum.lerp_(grad, weight=1.0 - group['lmbda'])

        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad

                self.state[p]['old_p'] = p.clone()
                self.state[f'old_grad_p_{i}']['old_grad_p'] = grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if 'old_p' in self.state[p]:
                    p.copy_(self.state[p].pop('old_p'))

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Closure = None):
        """Perturb weights, recompute gradients, and apply the base optimizer update.

        Args:
            closure: Callable that clears gradients and recomputes the loss and gradients. Compute the initial
                gradients before calling this method.

        Raises:
            NoClosureError: No closure is supplied.

        """
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['base_optimizer'] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if 'base_optimizer' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
            self.param_groups = self.base_optimizer.param_groups
        else:
            self.base_optimizer.param_groups = self.param_groups
