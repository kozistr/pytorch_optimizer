import math
from enum import IntEnum

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


class LMONorm(IntEnum):
    """Linear minimization oracle normalization types."""

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:
    """Identity linear minimization oracle and parameter initialization."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize a parameter tensor."""
        return x

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the linear minimization oracle direction."""
        return grad.clone()


class Col(Norm):
    """Column wise normalization for a linear minimization oracle.

    Args:
        normalized: Normalize by the input dimension. Use for non input layers.
        transpose: Transpose input before normalization. Use for embedding layers with shape (vocab_size,
            embedding_dim).

    """

    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=0, keepdim=True)).mul_(math.sqrt(x.size(0)))
        if self.normalized:
            x.div_(x.size(1))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        d_in, d_out = grad.size()

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True)) / math.sqrt(d_in)
        if self.normalized:
            rms_value.mul_(d_out)

        grad = grad / rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class Row(Norm):
    """Row wise normalization for a linear minimization oracle.

    Args:
        normalized: Normalize by the input dimension. Use for non input layers.
        transpose: Transpose input before normalization. Use for embedding layers with shape (vocab_size,
            embedding_dim).

    """

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=-1, keepdim=True))
        if self.normalized:
            x.div_(math.sqrt(x.size(-1)))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=-1, keepdim=True))
        if self.normalized:
            rms_value.mul_(math.sqrt(grad.size(-1)))

        grad = grad / rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class BiasRMS(Norm):
    """Root mean square normalization for bias parameters."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.zeros_(x)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms_value = torch.sqrt(torch.mean(grad.pow(2), dim=0, keepdim=True))
        return grad / rms_value.add_(eps)


class SpectralConv(Norm):
    """Spectral normalization for convolutional weights.

    Args:
        num_steps: Number of steps of zero power Newton-Schulz normalization, typically 5.

    """

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double().contiguous()

        d_out, d_in, *kernel_shape = x_fp64.size()
        kernel_volume = math.prod(kernel_shape)
        kernels = x_fp64.reshape(d_out, d_in, kernel_volume)
        for i in range(kernel_volume):
            torch.nn.init.orthogonal_(kernels[:, :, i])

        x_fp64.mul_(math.sqrt(d_out / d_in) / kernel_volume)

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)

        d_out, d_in, *kernel_shape = grad.size()

        grad *= math.sqrt(d_out / d_in) / math.prod(kernel_shape)

        return grad


class Spectral(Norm):
    """Spectral normalization.

    Args:
        max_scale: Set upper bound (1.0) of the scale.
        normalize: Normalize by the input dimension. Use for non input layers.
        num_steps: Number of zero power Newton-Schulz normalization steps, typically 5.

    """

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        torch.nn.init.orthogonal_(x_fp64)

        d_out, d_in = x_fp64.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        x_fp64.mul_(scale)

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)

        d_out, d_in = grad.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        grad *= scale

        return grad


class Sign(Norm):
    """Sign normalization.

    Args:
        zero_init: Initialize with zero.
        normalize: Normalize by the input dimension. Use for non input layers.

    """

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            return torch.nn.init.zeros_(x)

        d_in: int = x.size(1)

        x = 2 * torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) - 1
        if self.normalize:
            x.div_(d_in)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        d_in: int = grad.size(1)
        return torch.sign(grad).div_(d_in) if self.normalize else torch.sign(grad)


class Auto(Norm):
    """Choose a normalization from the parameter shape."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ndim: int = x.ndim
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        ndim: int = grad.ndim
        if ndim in (0, 1):
            return BiasRMS().lmo(grad)
        if ndim == 2:
            return Spectral().lmo(grad)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad)
        raise NotImplementedError


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    """Construct a linear minimization oracle normalization by type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()


class SCION(BaseOptimizer):
    """Norm constrained updates using linear minimization oracles.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Weight of the new gradient in the momentum average, equal to `1 - usual_momentum`.
        constraint: Use conditional gradient updates within the selected norm radius.
        norm_type: Linear minimization oracle type, as an `LMONorm` value or matching integer.
        norm_kwargs: Options for the normalization class.
        scale: Radius or update scale for the selected normalization.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    Examples:
        ```python
        from pytorch_optimizer import SCION
        from pytorch_optimizer.optimizer.scion import LMONorm

        radius = 50.0
        parameter_groups = [{
            'params': model.transformer.h.parameters(),
            'norm_type': LMONorm.SPECTRAL,
            'scale': radius,
        }, {
            'params': model.lm_head.parameters(),
            'norm_type': LMONorm.SIGN,
            'scale': radius * 60.0,
        }]
        optimizer = SCION(parameter_groups)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        norm_type: int = LMONorm.AUTO,
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')

        self.foreach = foreach
        self.maximize = maximize

        if norm_kwargs is None:
            norm_kwargs = {}

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'norm_type': norm_type,
            'norm_kwargs': norm_kwargs,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCION'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if 'd' not in state:
                state['d'] = torch.zeros_like(grad)

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])
            for p in group['params']:
                p.copy_(norm.init(p))
                p.mul_(group['scale'])

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        norm: Norm,
        ds: list[torch.Tensor],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        if not group['constraint'] and group['weight_decay'] > 0.0:
            self.apply_weight_decay_foreach(
                params,
                grads=grads,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

        torch._foreach_lerp_(ds, grads, group['momentum'])

        updates = [norm.lmo(d) for d in ds]
        torch._foreach_mul_(updates, group['scale'])

        if group['constraint']:
            torch._foreach_mul_(params, 1.0 - group['lr'])

        torch._foreach_add_(params, updates, alpha=-group['lr'])

    def _step_per_param(self, group: ParamGroup, norm: Norm) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            d = state['d']

            if not group['constraint'] and group['weight_decay'] > 0.0:
                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

            d.lerp_(grad, weight=group['momentum'])

            update = norm.lmo(d).mul_(group['scale'])

            if group['constraint']:
                p.mul_(1.0 - group['lr'])

            p.add_(update, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=['d'])
                if params:
                    self._step_foreach(group, params, grads, norm, state_dict['d'])
            else:
                self._step_per_param(group, norm)

        return loss


class SCIONLight(BaseOptimizer):
    """Variant of SCION that stores momentum in gradient buffers.

    Reuse gradient buffers between backward passes to retain momentum. Each `step()`
    scales those buffers by `1 - momentum` after updating parameters.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Complement of the factor retained in gradient buffers after each update.
        constraint: Use conditional gradient updates within the selected norm radius.
        norm_type: Linear minimization oracle type, as an `LMONorm` value or matching integer.
        norm_kwargs: Options for the normalization class.
        scale: Radius or update scale for the selected normalization.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    Examples:
        ```python
        from pytorch_optimizer import SCIONLight
        from pytorch_optimizer.optimizer.scion import LMONorm

        radius = 50.0
        parameter_groups = [{
            'params': model.transformer.h.parameters(),
            'norm_type': LMONorm.SPECTRAL,
            'scale': radius,
        }, {
            'params': model.lm_head.parameters(),
            'norm_type': LMONorm.SIGN,
            'scale': radius * 60.0,
        }]
        optimizer = SCIONLight(parameter_groups)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        norm_type: int = LMONorm.AUTO,
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')

        self.foreach = foreach
        self.maximize = maximize

        if norm_kwargs is None:
            norm_kwargs = {}

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'norm_type': norm_type,
            'norm_kwargs': norm_kwargs,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCIONLight'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])
            for p in group['params']:
                p.copy_(norm.init(p))
                p.mul_(group['scale'])

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        if group.get('foreach') is False:
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        norm: Norm,
    ) -> None:
        momentum = group['momentum']

        if self.maximize:
            torch._foreach_neg_(grads)

        updates = [norm.lmo(grad) for grad in grads]
        torch._foreach_mul_(updates, group['scale'])

        if group['constraint']:
            torch._foreach_mul_(params, 1.0 - group['lr'])

        if not group['constraint'] and group['weight_decay'] > 0.0:
            self.apply_weight_decay_foreach(
                params,
                grads=grads,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

        torch._foreach_add_(params, updates, alpha=-group['lr'])

        if momentum != 1.0:
            torch._foreach_mul_(grads, 1.0 - momentum)

    def _step_per_param(self, group: ParamGroup, norm: Norm) -> None:
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            self.maximize_gradient(grad, maximize=self.maximize)

            update = norm.lmo(grad).mul_(group['scale'])

            if group['constraint']:
                p.mul_(1.0 - group['lr'])

            if not group['constraint'] and group['weight_decay'] > 0.0:
                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

            p.add_(update, alpha=-group['lr'])

            if momentum != 1.0:
                grad.mul_(1.0 - momentum)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            norm = build_lmo_norm(group['norm_type'], **group['norm_kwargs'])

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                if params:
                    self._step_foreach(group, params, grads, norm)
            else:
                self._step_per_param(group, norm)

        return loss
