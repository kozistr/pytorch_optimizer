import math
from typing import cast

import torch
from torch import nn
from torch.distributed import all_gather, get_rank, get_world_size
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    foreach_add_,
    foreach_addcdiv_,
    group_tensors_by_device_and_dtype,
)
from pytorch_optimizer.optimizer.utils.matrix import (
    NewtonSchulzWeights,
    get_newton_schulz_weights,
    zero_power_via_newton_schulz_5,
)


def get_adjusted_lr(
    lr: float | torch.Tensor, param_shape: tuple[int, ...], use_adjusted_lr: bool = False
) -> float | torch.Tensor:
    """Scale the learning rate for an orthogonal matrix update.

    Args:
        lr: Base learning rate.
        param_shape: Weight shape, with output features first and remaining dimensions as input features.
        use_adjusted_lr: Use the Moonlight factor `sqrt(max(1, output / input))`. Otherwise use `0.2 *
            sqrt(max(output, input))`.

    Returns:
        float | torch.Tensor: Shape adjusted learning rate.

    """
    output_shape, *input_shape = param_shape
    input_shape = math.prod(input_shape)

    ratio: float = (
        math.pow(max(1.0, output_shape / input_shape), 0.5)
        if use_adjusted_lr
        else 0.2 * math.sqrt(max(output_shape, input_shape))
    )

    return lr * ratio


class MuonBase(BaseOptimizer):
    """Shared batched updates for local Muon optimizers."""

    _supports_compiled_foreach = True
    _muon_state_keys: tuple[str, ...]

    def _step_foreach_group(self, group: ParamGroup) -> None:
        keys = list(self._muon_state_keys) if group['use_muon'] else ['exp_avg', 'exp_avg_sq']
        params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=keys)

        beta1, beta2 = group.get('betas', (0.0, 0.0))
        bias_correction1 = self.debias(beta1, group['step'])
        bias_correction2 = self.debias(beta2, group['step'])

        for tensors in group_tensors_by_device_and_dtype(params, grads, state_dict):
            if group['use_muon']:
                shapes: dict[tuple[int, int], list[int]] = {}
                for index, p in enumerate(tensors['params']):
                    rows, columns = p.size(0), p.numel() // p.size(0)
                    shape = (rows, columns) if group['use_adjusted_lr'] else (min(rows, columns), max(rows, columns))
                    shapes.setdefault(shape, []).append(index)

                batches = [tensors] if len(shapes) == 1 else [
                    {key: [values[index] for index in indices] for key, values in tensors.items() if key != 'indices'}
                    for indices in shapes.values()
                ]
            else:
                batches = [tensors]

            for batch in batches:
                self._step_foreach(
                    group,
                    batch['params'],
                    batch['grads'],
                    {key: batch[key] for key in keys},
                    bias_correction1,
                    bias_correction2,
                )

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction1: float | torch.Tensor,
        bias_correction2: float | torch.Tensor,
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], fixed_decay=False
        )

        if group['use_muon']:
            self._step_muon_foreach(group, params, grads, state_dict, bias_correction2)
        else:
            beta1, beta2 = group['betas']
            exp_avgs, exp_avg_sqs = state_dict['exp_avg'], state_dict['exp_avg_sq']

            torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta1)
            torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), weight=1.0 - beta2)

            de_noms = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(de_noms, group['eps'])
            torch._foreach_div_(de_noms, bias_correction2**0.5)

            foreach_addcdiv_(params, torch._foreach_div(exp_avgs, bias_correction1), de_noms, -group['lr'])

    @staticmethod
    def _orthogonalize(group: ParamGroup, updates: list[torch.Tensor]) -> list[torch.Tensor]:
        matrices = [update.reshape(update.size(0), -1) for update in updates]
        transposed = [matrix.size(0) > matrix.size(1) for matrix in matrices]
        matrix = matrices[0] if len(matrices) == 1 else torch.stack([
            matrix.mT if transpose else matrix for matrix, transpose in zip(matrices, transposed)
        ])

        result = zero_power_via_newton_schulz_5(matrix, num_steps=group['ns_steps'], weights=group['ns_coeffs'])

        return [result] if len(matrices) == 1 else [
            update.mT if transpose else update for update, transpose in zip(result.unbind(), transposed)
        ]

    def _momentum_updates(
        self, group: ParamGroup, grads: list[torch.Tensor], buffers: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        torch._foreach_lerp_(buffers, grads, weight=1.0 - group['momentum'])

        if group['nesterov']:
            torch._foreach_lerp_(grads, buffers, weight=group['momentum'])
            return self._orthogonalize(group, grads)

        return self._orthogonalize(group, buffers)

    def _apply_muon_updates(
        self, group: ParamGroup, params: list[torch.Tensor], grads: list[torch.Tensor], updates: list[torch.Tensor]
    ) -> None:
        updates = [update.reshape(p.shape) for p, update in zip(params, updates)]

        if group.get('cautious'):
            for update, grad in zip(updates, grads):
                self.apply_cautious(update, grad)

        lr = get_adjusted_lr(group['lr'], params[0].shape, use_adjusted_lr=group['use_adjusted_lr'])
        foreach_add_(params, updates, alpha=-lr)


class Muon(MuonBase):
    """Momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        use_adjusted_lr: Scale orthogonal updates using the Moonlight shape adjustment.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes. `False` disables batching; `None` enables it.

    Examples:
        ```python
        from pytorch_optimizer import Muon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = Muon(param_groups)
        ```

    """

    _muon_state_keys = ('momentum_buffer',)

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize
        self.foreach = foreach

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, {'foreach': foreach, **kwargs})

    def __str__(self) -> str:
        return 'Muon'

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

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_muon_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        updates = self._momentum_updates(group, grads, state_dict['momentum_buffer'])
        self._apply_muon_updates(group, params, grads, updates)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                self._step_foreach_group(group)
                continue

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

                if group['use_muon']:
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    )

                    if group.get('cautious'):
                        self.apply_cautious(update.reshape(p.shape), grad)

                    lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class DistributedMuon(BaseOptimizer):  # pragma: no cover
    """Distributed momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.
    Requires an initialized distributed process group.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        use_adjusted_lr: Scale orthogonal updates using the Moonlight shape adjustment.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        maximize: Maximize the objective instead of minimizing it.

    Examples:
        ```python
        from pytorch_optimizer import DistributedMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = DistributedMuon(param_groups)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize

        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'DistributedMuon'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0 and not group['use_muon']:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if group['use_muon']:
                params = group['params']
                padded_params = params + [torch.empty_like(params[-1])] * (
                    self.world_size - len(params) % self.world_size
                )

                for i in range(len(params))[:: self.world_size]:
                    if i + self.rank < len(params):
                        p = params[i + self.rank]

                        grad = p.grad

                        self.maximize_gradient(grad, maximize=self.maximize)

                        state = self.state[p]
                        if len(state) == 0:
                            state['momentum_buffer'] = torch.zeros_like(p)

                        self.apply_weight_decay(
                            p,
                            grad=grad,
                            lr=group['lr'],
                            weight_decay=group['weight_decay'],
                            weight_decouple=group['weight_decouple'],
                            fixed_decay=False,
                        )

                        buf = state['momentum_buffer']
                        buf.lerp_(grad, weight=1.0 - group['momentum'])

                        update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                        if update.ndim > 2:
                            update = update.view(len(update), -1)

                        update = zero_power_via_newton_schulz_5(
                            update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                        )

                        if group.get('cautious'):
                            self.apply_cautious(update, grad)

                        lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                        p.add_(update.reshape(p.shape), alpha=-lr)

                    all_gather(padded_params[i:i + self.world_size], padded_params[i + self.rank])  # fmt: skip
            else:
                for p in group['params']:
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
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class AdaMuon(MuonBase):
    """Adaptive momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        betas: Decay rates for gradient momentum and squared orthogonalized updates.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        use_adjusted_lr: Scale orthogonal updates using the Moonlight shape adjustment.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes. `False` disables batching; `None` enables it.

    Examples:
        ```python
        from pytorch_optimizer import AdaMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = AdaMuon(param_groups)
        ```

    """

    _muon_state_keys = ('m', 'v')

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        betas: Betas = (0.9, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.999),
        adamw_wd: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(betas)
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(eps, 'eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize
        self.foreach = foreach

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['betas'] = group.get('betas', betas)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)
            group['eps'] = group.get('eps', eps)

        super().__init__(params, {'foreach': foreach, **kwargs})

    def __str__(self) -> str:
        return 'AdaMuon'

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

            if len(state) == 0:
                if group['use_muon']:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p.flatten())
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_muon_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        beta1, beta2 = group['betas']
        moments, variances = state_dict['m'], state_dict['v']

        torch._foreach_lerp_(moments, grads, weight=1.0 - beta1)

        updates = [update.flatten() for update in self._orthogonalize(group, moments)]

        torch._foreach_mul_(variances, beta2)
        torch._foreach_addcmul_(variances, updates, updates, value=1.0 - beta2)

        de_noms = torch._foreach_sqrt(torch._foreach_div(variances, bias_correction2))
        torch._foreach_add_(de_noms, group['eps'])
        torch._foreach_div_(updates, de_noms)

        torch._foreach_mul_(updates, 0.2 * math.sqrt(params[0].numel()))
        norms = [update.norm().add_(group['eps']) for update in updates]
        torch._foreach_div_(updates, norms)

        updates = [update.reshape(p.shape) for p, update in zip(params, updates)]
        lr = get_adjusted_lr(group['lr'], params[0].shape, use_adjusted_lr=group['use_adjusted_lr'])
        foreach_add_(params, updates, alpha=-lr)

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                self._step_foreach_group(group)
                continue

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

                if group['use_muon']:
                    m = state['m']
                    m.lerp_(grad, weight=1.0 - beta1)

                    update = m

                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    ).flatten()

                    v = state['v']
                    v.mul_(beta2).addcmul_(update, update, value=1.0 - beta2)

                    update.div_((v / bias_correction2).sqrt_().add_(group['eps']))
                    update = update.reshape(p.size())

                    update.mul_(0.2 * math.sqrt(p.numel())).div_(update.norm().add_(group['eps']))

                    lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update, alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(math.sqrt(bias_correction2))

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class AdaGO(MuonBase):
    """Orthogonal momentum updates with AdaGrad step size adaptation.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        gamma: Gradient norm cap for the accumulator and adaptive step size.
        v: Initial value of the AdaGrad accumulator.
        eps: Epsilon value. Lower bound eps > 0 on the stepsizes.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        use_adjusted_lr: Scale orthogonal updates using the Moonlight shape adjustment.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes. `False` disables batching; `None` enables it.

    Examples:
        ```python
        from pytorch_optimizer import AdaGO

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = AdaGO(param_groups)
        ```

    """

    _muon_state_keys = ('momentum_buffer', 'v')

    def __init__(
        self,
        params: ParamsT,
        lr: float = 5e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        gamma: float = 10.0,
        eps: float = 5e-4,
        v: float = 1e-6,
        nesterov: bool = False,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_positive(gamma, 'gamma')
        self.validate_positive(eps, 'eps')
        self.validate_positive(v, 'v')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize
        self.foreach = foreach

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['gamma'] = group.get('gamma', gamma)
                group['eps'] = group.get('eps', eps)
                group['v'] = group.get('v', v)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, {'foreach': foreach, **kwargs})

    def __str__(self) -> str:
        return 'AdaGO'

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

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['v'] = torch.tensor(group['v'], dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_muon_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        buffers, variances = state_dict['momentum_buffer'], state_dict['v']

        torch._foreach_lerp_(buffers, grads, weight=1.0 - group['momentum'])

        grad_norms = torch._foreach_norm(grads, ord=2)
        squared_norms = torch._foreach_mul(grad_norms, grad_norms)
        torch._foreach_clamp_max_(squared_norms, group['gamma'] ** 2)
        torch._foreach_add_(variances, squared_norms)

        if group['nesterov']:
            torch._foreach_lerp_(grads, buffers, weight=group['momentum'])

        updates = self._orthogonalize(group, grads if group['nesterov'] else buffers)
        updates = [update.reshape(p.shape) for p, update in zip(params, updates)]

        if group.get('cautious'):
            for update, grad in zip(updates, grads):
                self.apply_cautious(update, grad)

        # Nesterov modifies gradients before the adaptive step size is computed.
        step_sizes = torch._foreach_norm(grads, ord=2) if group['nesterov'] else grad_norms
        torch._foreach_clamp_max_(step_sizes, group['gamma'])

        lr = get_adjusted_lr(group['lr'], params[0].shape, use_adjusted_lr=group['use_adjusted_lr'])
        torch._foreach_mul_(step_sizes, lr)
        torch._foreach_div_(step_sizes, variances)
        torch._foreach_clamp_min_(step_sizes, group['eps'])

        torch._foreach_addcmul_(params, [update.to(params[0].dtype) for update in updates], step_sizes, value=-1.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                self._step_foreach_group(group)
                continue

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

                if group['use_muon']:
                    buf, v = state['momentum_buffer'], state['v']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    grad_norm = grad.norm(p=2.0)
                    v.add_(grad_norm.square().clamp_max_(group['gamma'] ** 2))

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    )

                    if group.get('cautious'):
                        self.apply_cautious(update.reshape(p.shape), grad)

                    lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    step_size = (lr * grad.norm(2).clamp_max_(group['gamma']) / v).clamp_min_(group['eps'])
                    p.addcmul_(update.reshape(p.shape).to(p.dtype), step_size, value=-1.0)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class NorMuon(MuonBase):
    """Muon updates with row wise second moment normalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        beta2: Decay rate of the row wise second moment of the orthogonalized update.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        update_scale: How to rescale the row normalized update. `preserve_norm` keeps the Frobenius norm of the
            orthogonalized update, as the official code does. `match_rms` gives it the Frobenius norm `0.2 * sqrt(m
            * n)`, so the RMS is 0.2 as in Algorithm 1 of the paper.
        use_adjusted_lr: Apply the Moonlight shape adjustment in `preserve_norm` mode. Unused in `match_rms` mode.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        eps: Term added to the denominator of the row wise normalization.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes. `False` disables batching; `None` enables it.

    Examples:
        ```python
        from pytorch_optimizer import NorMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = NorMuon(param_groups)
        ```

    """

    _muon_state_keys = ('momentum_buffer', 'second_momentum_buffer')

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        update_scale: str = 'preserve_norm',
        use_adjusted_lr: bool = True,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta2, 'beta2', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_options(update_scale, 'update_scale', ['preserve_norm', 'match_rms'])
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        self.validate_non_negative(eps, 'eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize
        self.foreach = foreach

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['beta2'] = group.get('beta2', beta2)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['update_scale'] = group.get('update_scale', update_scale)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
                group['eps'] = group.get('eps', eps)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, {'foreach': foreach, **kwargs})

    def __str__(self) -> str:
        return 'NorMuon'

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

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['second_momentum_buffer'] = p.new_zeros(p.size(0), 1)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_muon_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        updates = self._momentum_updates(group, grads, state_dict['momentum_buffer'])
        updates = [update.to(grads[0].dtype) for update in updates]
        original_norms = torch._foreach_norm(updates, ord=2)

        second_moments = state_dict['second_momentum_buffer']
        row_means = [update.square().mean(dim=-1, keepdim=True) for update in updates]
        torch._foreach_lerp_(second_moments, row_means, weight=1.0 - group['beta2'])

        de_noms = torch._foreach_sqrt(second_moments)
        torch._foreach_add_(de_noms, group['eps'])
        torch._foreach_div_(updates, de_noms)

        norms = torch._foreach_norm(updates, ord=2)
        torch._foreach_add_(norms, group['eps'])

        if group['update_scale'] == 'preserve_norm':
            torch._foreach_mul_(updates, torch._foreach_div(original_norms, norms))
            lr = get_adjusted_lr(group['lr'], params[0].shape, use_adjusted_lr=group['use_adjusted_lr'])
        else:
            scales = [0.2 * math.sqrt(update.numel()) / norm for update, norm in zip(updates, norms)]
            torch._foreach_mul_(updates, scales)
            lr = group['lr']

        updates = [update.reshape(p.shape) for p, update in zip(params, updates)]
        foreach_add_(params, updates, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                self._step_foreach_group(group)
                continue

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

                if group['use_muon']:
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    update = update.reshape(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    ).to(grad.dtype)

                    original_norm = update.norm()

                    v_mean = update.square().mean(dim=-1, keepdim=True)
                    second_momentum = state['second_momentum_buffer']
                    second_momentum.lerp_(v_mean, weight=1.0 - group['beta2'])

                    update.div_(second_momentum.sqrt().add_(group['eps']))

                    if group['update_scale'] == 'preserve_norm':
                        update.mul_(original_norm / update.norm().add_(group['eps']))
                        lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])
                    else:
                        update.mul_(0.2 * math.sqrt(update.numel()) / update.norm().add_(group['eps']))
                        lr = group['lr']

                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


def prepare_muon_parameters(
    model: nn.Module,
    optimizer_name: str,
    lr: float | torch.Tensor,
    weight_decay: float,
    adamw_lr: float = 3e-4,
    adamw_wd: float = 0.0,
    **kwargs,
) -> Optimizer:
    """Create a Muon family optimizer by grouping model parameters.

    Classifies weights by parameter name and dimensionality. Review the resulting groups,
    or construct them yourself using the optimizer's example for model specific control.

    Args:
        model: Model whose parameters to group.
        optimizer_name: Muon family optimizer name.
        lr: Learning rate for the orthogonal update groups.
        weight_decay: Weight decay for the orthogonal update groups.
        adamw_lr: Learning rate for the AdamW groups.
        adamw_wd: Weight decay for the AdamW groups.
        **kwargs (dict): Options for the selected optimizer.

    Returns:
        Optimizer: Optimizer with orthogonal update and AdamW parameter groups.

    """
    muon_parameters: dict[int, torch.Tensor] = {}
    non_muon_params: dict[int, torch.Tensor] = {}

    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f'{module_name}.{name}' if module_name else name
            if (
                isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM, nn.Conv2d))
                and param.ndim >= 2
                and 'head' not in full_name
            ):
                muon_parameters[id(param)] = param
            else:
                non_muon_params[id(param)] = param

    param_groups: ParamsT = [
        {
            'params': [p for key, p in muon_parameters.items() if key not in non_muon_params],
            'lr': lr,
            'weight_decay': weight_decay,
            'use_muon': True,
        },
        {'params': list(non_muon_params.values()), 'lr': adamw_lr, 'weight_decay': adamw_wd, 'use_muon': False},
    ]

    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adamuon':
        return AdaMuon(param_groups, **kwargs)
    if optimizer_name == 'adago':
        return AdaGO(param_groups, **kwargs)
    if optimizer_name == 'normuon':
        return NorMuon(param_groups, **kwargs)

    return Muon(param_groups, **kwargs)
