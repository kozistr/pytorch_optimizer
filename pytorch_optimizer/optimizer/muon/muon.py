import math
from typing import Literal, cast

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon.base import MuonBase, get_adjusted_lr
from pytorch_optimizer.optimizer.utils.matrix import NewtonSchulzWeights, get_newton_schulz_weights


class Muon(MuonBase):
    """Momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.
    Supports FSDP2 DTensor parameters with eager optimizer steps.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        ns_method: Orthogonalization method, `newton_schulz` or `gram_newton_schulz`.
        ns_reset_steps: Iteration counts after which to restart Gram Newton-Schulz.
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
        foreach: bool | None = None,
        ns_method: Literal['newton_schulz', 'gram_newton_schulz'] = 'newton_schulz',
        ns_reset_steps: tuple[int, ...] = (2,),
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_options(ns_method, 'ns_method', ['newton_schulz', 'gram_newton_schulz'])
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
                group['ns_method'] = group.get('ns_method', ns_method)
                self.validate_options(group['ns_method'], 'ns_method', ['newton_schulz', 'gram_newton_schulz'])
                group['ns_reset_steps'] = group.get('ns_reset_steps', ns_reset_steps)
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
                    update = self._momentum_updates(group, [grad], [buf])[0]

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

                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                    p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)

        return loss
