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
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype
from pytorch_optimizer.optimizer.utils.gradient import get_global_gradient_norm
from pytorch_optimizer.optimizer.utils.model import disable_running_stats, enable_running_stats


def _get_perturbations_foreach(
    params: list[torch.Tensor], grads: list[torch.Tensor], scale: torch.Tensor, adaptive: bool
) -> tuple[torch.Tensor, ...]:
    if adaptive:
        updates = torch._foreach_pow(params, 2)
        torch._foreach_mul_(updates, grads)
    else:
        updates = torch._foreach_mul(grads, 1.0)

    torch._foreach_mul_(updates, scale.to(params[0]))

    return updates


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
        foreach: Batch FP32/FP64 perturbations and restoration. `None` enables batching on CUDA.
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
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        if foreach is not None:
            kwargs['foreach'] = foreach

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive, 'foreach': foreach, **kwargs}

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
        grad_norm = get_global_gradient_norm(self.param_groups, weight_adaptive=True)
        grad_norm.sqrt_().squeeze_(0).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            if self.can_use_foreach(
                group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
            ):
                params, grads, _ = self.collect_trainable_params(group, self.state)

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    if self.use_gc:
                        centralize_gradient(batch['grads'], gc_conv_only=False)

                    for p in batch['params']:
                        self.state[p]['old_p'] = p.clone()

                    updates = _get_perturbations_foreach(batch['params'], batch['grads'], scale, group['adaptive'])
                    torch._foreach_add_(batch['params'], updates)

                continue

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
            if self.can_use_foreach(
                group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
            ):
                params = [p for p in group['params'] if 'old_p' in self.state[p]]
                saved = [self.state[p].pop('old_p') for p in params]

                for batch in group_tensors_by_device_and_dtype(params, saved):
                    torch._foreach_copy_(batch['params'], batch['grads'])

                continue

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
        foreach: Batch perturbations and gradient decomposition. `None` enables batching on CUDA.
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
        foreach: bool | None = None,
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

        self.base_optimizer = base_optimizer

        defaults: Defaults = {'adaptive': adaptive, 'foreach': foreach, **kwargs}

        super().__init__(params, defaults)
        self.param_groups = self.base_optimizer.param_groups

        for group in self.param_groups:
            if foreach is not None:
                group['foreach'] = foreach

        self.update_rho_t()

    def __str__(self) -> str:
        return 'GSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return self.can_use_foreach(
            group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
        )

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
                self.state[p].pop('old_g', None)

                if p.grad is not None:
                    self.state[p]['old_g'] = p.grad.clone()

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    updates = _get_perturbations_foreach(batch['params'], batch['grads'], scale, self.adaptive)
                    torch._foreach_add_(batch['params'], updates)

                    for p, update in zip(batch['params'], updates):
                        self.state[p]['e_w'] = update

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def un_perturb(self):
        for group in self.param_groups:
            if self._can_use_foreach(group):
                params = [p for p in group['params'] if 'e_w' in self.state[p]]
                updates = [self.state[p].pop('e_w') for p in params]

                for batch in group_tensors_by_device_and_dtype(params, updates):
                    torch._foreach_sub_(batch['params'], batch['grads'])

                continue

            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.sub_(self.state[p].pop('e_w'))

    @torch.no_grad()
    def gradient_decompose(self, alpha: float = 0.0):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if 'old_g' not in self.state[p]:
                    self.state[p]['old_g'] = torch.zeros_like(p)

                inner_prod += torch.sum(self.state[p]['old_g'] * p.grad)

        new_grad_norm = self.grad_norm(by=None)
        old_grad_norm = self.grad_norm(by='old_g')

        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for group in self.param_groups:
            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(group, self.state, ['old_g'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    scale = cosine * old_grad_norm / (new_grad_norm + self.perturb_eps)
                    verticals = torch._foreach_mul(batch['grads'], -scale.to(batch['grads'][0]))
                    torch._foreach_add_(verticals, batch['old_g'])
                    torch._foreach_add_(batch['grads'], verticals, alpha=-alpha)

                continue

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

                    all_reduce(p.grad, op=ReduceOp.SUM)
                    p.grad.div_(float(get_world_size()))

    @torch.no_grad()
    def grad_norm(self, by: str | None = None, weight_adaptive: bool = False) -> torch.Tensor:
        if not by and not weight_adaptive:
            return get_global_gradient_norm(self.param_groups).sqrt_().squeeze(0)

        norms = [
            ((torch.abs(p) if weight_adaptive else 1.0) * (p.grad if not by else self.state[p][by])).norm(p=2)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]

        if not norms:
            return get_global_gradient_norm(self.param_groups).sqrt_().squeeze(0)

        return torch.stack(norms).norm(p=2)

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
        foreach: Batch FP32/FP64 perturbations and gradient mixing when `True`. `None` uses scalar wrapper updates.
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
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')

        self.model = model
        self.decouple = decouple
        self.max_norm = max_norm

        alpha: float = gamma / (1.0 - gamma)

        if foreach is not None:
            kwargs['foreach'] = foreach

        defaults: Defaults = {
            'rho': rho,
            'alpha': alpha,
            'adaptive': adaptive,
            'sam_eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'WSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return not is_initialized() and self.can_use_foreach(
            group, group.get('foreach') is True, dtypes=(torch.float32, torch.float64)
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = get_global_gradient_norm(self.param_groups, weight_adaptive=True).sqrt_().squeeze_(0)

        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + group['sam_eps'])

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    updates = _get_perturbations_foreach(batch['params'], batch['grads'], scale, group['adaptive'])
                    torch._foreach_add_(batch['params'], updates)

                    for p, update in zip(batch['params'], updates):
                        self.state[p]['e_w'] = update

                continue

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

    def _mix_gradients(self, group: ParamGroup) -> None:
        if self._can_use_foreach(group):
            params, grads, old_grads = [], [], []

            for p in group['params']:
                old_grad = self.state[p].pop('grad', None)

                if p.grad is None:
                    continue

                params.append(p)
                grads.append(p.grad)
                old_grads.append(torch.zeros_like(p.grad) if old_grad is None else old_grad)

            for batch in group_tensors_by_device_and_dtype(params, grads, {'old_grads': old_grads}):
                if self.decouple:
                    sharpness = torch._foreach_sub(batch['grads'], batch['old_grads'])

                    for p, update in zip(batch['params'], sharpness):
                        self.state[p]['sharpness'] = update

                    torch._foreach_copy_(batch['grads'], batch['old_grads'])
                else:
                    torch._foreach_lerp_(batch['grads'], batch['old_grads'], weight=1.0 - group['alpha'])

            return

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

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            if self._can_use_foreach(group):
                params = [p for p in group['params'] if 'e_w' in self.state[p]]
                perturbations = [self.state[p].pop('e_w') for p in params]

                for batch in group_tensors_by_device_and_dtype(params, perturbations):
                    torch._foreach_sub_(batch['params'], batch['grads'])

                continue

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
            self._mix_gradients(group)

        self.base_optimizer.step()

        if self.decouple:
            for group in self.param_groups:
                if self._can_use_foreach(group):
                    params, grads, states = self.collect_trainable_params(group, self.state, ['sharpness'])

                    for batch in group_tensors_by_device_and_dtype(params, grads, states):
                        foreach_add_(batch['params'], batch['sharpness'], alpha=-group['lr'] * group['alpha'])

                    continue

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
        foreach: Batch perturbations and moment updates in eager execution. `None` enables batching when supported.
        **kwargs (dict): Parameters for optimizer.

    """

    def __init__(
        self,
        params: ParamsT,
        num_data: int,
        lr: float | torch.Tensor = 5e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-4,
        rho: float = 0.05,
        adaptive: bool = False,
        damping: float = 0.1,
        foreach: bool | None = None,
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
            'foreach': foreach,
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

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        compiling = torch._dynamo.is_compiling()  # ty: ignore[deprecated]
        return not compiling and self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        perturb: bool,
    ) -> None:
        if perturb:
            if group['adaptive']:
                updates = torch._foreach_pow(params, 2)
                torch._foreach_mul_(updates, group['rho'])
                torch._foreach_mul_(updates, grads)
            else:
                updates = torch._foreach_mul(grads, group['rho'])

            torch._foreach_div_(updates, state['s'])
            torch._foreach_add_(params, updates)
            return

        beta1, beta2 = group['betas']
        scaled_grads = torch._foreach_mul(grads, group['weight_decay'])
        torch._foreach_lerp_(state['momentum'], scaled_grads, weight=1.0 - beta1)

        variances = torch._foreach_sqrt(state['s'])
        torch._foreach_mul_(variances, torch._foreach_abs(grads))
        torch._foreach_add_(variances, group['weight_decay'] + self.damping)
        torch._foreach_pow_(variances, 2)
        torch._foreach_lerp_(state['s'], variances, weight=1.0 - beta2)

        foreach_add_(params, torch._foreach_div(state['momentum'], state['s']), alpha=-group['lr'])

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group):
                params, _, state = self.collect_trainable_params(group, self.state, ['s'])
                noises = [torch.normal(0.0, 1 / (self.num_data * value)) for value in state['s']]
                for batch in group_tensors_by_device_and_dtype(params, noises):
                    torch._foreach_add_(batch['params'], batch['grads'])

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                noise = torch.normal(0.0, 1 / (self.num_data * state['s']))

                p.add_(noise)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(group, self.state, ['s'])
                for p, grad in zip(params, grads):
                    self.state[p]['noisy_gradient'] = grad.clone()

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, True)

                continue

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
            if self._can_use_foreach(group):
                params, grads, state = self.collect_trainable_params(group, self.state, ['s', 'momentum'])
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, False)

                continue

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
        foreach: Batch perturbations and gradient projections. `None` enables batching on CUDA.
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
        foreach: bool | None = None,
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

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive, 'foreach': foreach}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'LookSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def get_step(self):
        group = self.param_groups[0]
        return group.get(
            'looksam_step', group.get('step', next(iter(self.base_optimizer.state.values()), {}).get('step', 0))
        )

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return self.can_use_foreach(
            group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        if self.get_step() % self.k != 0:
            if zero_grad:
                self.zero_grad()

            return

        grad_norm = get_global_gradient_norm(self.param_groups, weight_adaptive=True)
        grad_norm.sqrt_().squeeze_(0).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            if self._can_use_foreach(group):
                for p in group['params']:
                    self.state[p].pop('old_grad_p', None)

                params, grads, _ = self.collect_trainable_params(group, self.state)

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    if self.use_gc:
                        centralize_gradient(batch['grads'], gc_conv_only=False)

                    for p, grad in zip(batch['params'], batch['grads']):
                        self.state[p]['old_p'] = p.clone()
                        self.state[p]['old_grad_p'] = grad.clone()

                    updates = _get_perturbations_foreach(batch['params'], batch['grads'], scale, group['adaptive'])
                    torch._foreach_add_(batch['params'], updates)

                continue

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
            if self._can_use_foreach(group):
                self._second_step_foreach(group, step)
                continue

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

        for group in self.param_groups:
            group['looksam_step'] = step + 1

        if zero_grad:
            self.zero_grad()

    def _second_step_foreach(self, group: ParamGroup, step: int) -> None:
        saved_params = [p for p in group['params'] if 'old_p' in self.state[p]]
        saved = [self.state[p].pop('old_p') for p in saved_params]

        for batch in group_tensors_by_device_and_dtype(saved_params, saved):
            torch._foreach_copy_(batch['params'], batch['grads'])

        params, grads, previous = [], [], []
        full_step = step % self.k == 0

        for p in group['params']:
            old_grad = self.state[p].pop('old_grad_p', None)
            previous_grad = old_grad if full_step else self.state[p].get('gv')

            if p.grad is not None and previous_grad is not None:
                params.append(p)
                grads.append(p.grad)
                previous.append(previous_grad)

        for batch in group_tensors_by_device_and_dtype(params, grads, {'previous': previous}):
            norms = torch._foreach_norm(batch['grads'])
            previous_norms = torch._foreach_norm(batch['previous'])

            if full_step:
                previous_units = torch._foreach_div(
                    batch['previous'], [norm.clamp_min(self.perturb_eps) for norm in previous_norms]
                )
                units = torch._foreach_div(batch['grads'], [norm.clamp_min(self.perturb_eps) for norm in norms])
                cosines = [product.sum() for product in torch._foreach_mul(previous_units, units)]
                projections = torch._foreach_mul(previous_units, torch._foreach_mul(norms, cosines))

                for p, gv in zip(batch['params'], torch._foreach_sub(batch['grads'], projections)):
                    self.state[p]['gv'] = gv
            else:
                scales = torch._foreach_div(norms, torch._foreach_add(previous_norms, 1e-8))
                updates = torch._foreach_mul(batch['previous'], scales)
                torch._foreach_add_(batch['grads'], updates, alpha=self.alpha)

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
        foreach: Batch momentum, perturbation, and restoration operations. `None` enables batching on CUDA.
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
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(sigma, 'sigma')
        self.validate_non_negative(lmbda, 'lmbda')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.perturb_eps = perturb_eps

        defaults: Defaults = {
            'rho': rho, 'sigma': sigma, 'lmbda': lmbda, 'adaptive': adaptive, 'foreach': foreach,
        }
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'FriendlySAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        return self.can_use_foreach(
            group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            if self._can_use_foreach(group):
                self._update_momentum_foreach(group)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'momentum' not in state:
                    state['momentum'] = grad.clone()
                else:
                    momentum = state['momentum']

                    raw_grad = grad.clone()
                    grad.sub_(momentum, alpha=group['sigma'])
                    momentum.lerp_(raw_grad, weight=1.0 - group['lmbda'])

        grad_norm = get_global_gradient_norm(self.param_groups, weight_adaptive=True)
        grad_norm.sqrt_().squeeze_(0).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)

                for p in params:
                    self.state[p]['old_p'] = p.clone()

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    updates = _get_perturbations_foreach(batch['params'], batch['grads'], scale, group['adaptive'])
                    torch._foreach_add_(batch['params'], updates)

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.state[p]['old_p'] = p.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    def _update_momentum_foreach(self, group: ParamGroup) -> None:
        ready = []

        for p in group['params']:
            if p.grad is None:
                continue

            if 'momentum' in self.state[p]:
                ready.append(p)
            else:
                self.state[p]['momentum'] = p.grad.clone()

        params, grads, state = self.collect_trainable_params({**group, 'params': ready}, self.state, ['momentum'])

        for batch in group_tensors_by_device_and_dtype(params, grads, state):
            if group['sigma'] == 1.0 and group['lmbda'] >= 0.5:
                torch._foreach_sub_(batch['grads'], batch['momentum'])
                torch._foreach_add_(batch['momentum'], batch['grads'], alpha=1.0 - group['lmbda'])
                continue

            raw_grads = [grad.clone() for grad in batch['grads']]
            torch._foreach_sub_(batch['grads'], batch['momentum'], alpha=group['sigma'])
            torch._foreach_lerp_(batch['momentum'], raw_grads, weight=1.0 - group['lmbda'])

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            if self._can_use_foreach(group):
                params = [p for p in group['params'] if 'old_p' in self.state[p]]
                saved = [self.state[p].pop('old_p') for p in params]

                for batch in group_tensors_by_device_and_dtype(params, saved):
                    torch._foreach_copy_(batch['params'], batch['grads'])

                continue

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
