import math
from typing import cast

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon.base import MuonBase, get_adjusted_lr
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_
from pytorch_optimizer.optimizer.utils.matrix import (
    DTensor,
    NewtonSchulzWeights,
    get_newton_schulz_weights,
    has_dtensor,
    zero_power_via_newton_schulz_5,
)


class AdaMuon(MuonBase):
    """Adaptive momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.
    Supports FSDP2 DTensor parameters with eager optimizer steps.

    The default shape scaling gives the adaptive update an RMS of 0.2 before multiplication
    by the learning rate. `use_adjusted_lr=False` selects Keller Jordan's scaling instead.

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
        use_adjusted_lr: bool = True,
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
                    state['v'] = torch.zeros_like(p if has_dtensor and isinstance(p, DTensor) else p.flatten())
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @staticmethod
    def _update_norm(update: torch.Tensor, eps: float) -> torch.Tensor:
        if has_dtensor and isinstance(update, DTensor):
            norm = update.norm(dtype=torch.float32).full_tensor().to(update.dtype)
            norm = DTensor.from_local(norm, device_mesh=update.device_mesh)
        else:
            norm = update.norm()
        return norm.add_(eps)

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

        updates = [
            update.reshape(variance.shape)
            for update, variance in zip(self._orthogonalize(group, moments), variances)
        ]

        torch._foreach_mul_(variances, beta2)
        torch._foreach_addcmul_(variances, updates, updates, value=1.0 - beta2)

        de_noms = torch._foreach_sqrt(torch._foreach_div(variances, bias_correction2))
        torch._foreach_add_(de_noms, group['eps'])
        torch._foreach_div_(updates, de_noms)

        norms = [self._update_norm(update, group['eps']) for update in updates]
        rows = params[0].size(0)
        torch._foreach_mul_(updates, math.sqrt(min(rows, params[0].numel() // rows)))
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
                    m, v = state['m'], state['v']
                    m.lerp_(grad, weight=1.0 - beta1)

                    update = m

                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    ).reshape(v.shape)

                    v.mul_(beta2).addcmul_(update, update, value=1.0 - beta2)

                    update.div_((v / bias_correction2).sqrt_().add_(group['eps']))
                    update = update.reshape(p.size())

                    scale = math.sqrt(min(p.size(0), p.numel() // p.size(0)))
                    update.mul_(scale / self._update_norm(update, group['eps']))

                    lr = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update, alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group['eps'])

                    p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)

        return loss
