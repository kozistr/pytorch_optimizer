import math
from collections import defaultdict

import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import ParamGroup
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, foreach_addcdiv_, group_tensors_by_device_and_dtype
from pytorch_optimizer.optimizer.utils.matrix import (
    DTensor,
    has_dtensor,
    zero_power_via_gram_newton_schulz,
    zero_power_via_newton_schulz_5,
)


def get_adjusted_lr(
    lr: float | torch.Tensor, param_shape: tuple[int, ...], use_adjusted_lr: bool = False
) -> float | torch.Tensor:
    """Scale the learning rate for an orthogonal matrix update.

    Args:
        lr: Base learning rate.
        param_shape: Weight shape, with output features first and remaining dimensions as input features.
        use_adjusted_lr: Use the Moonlight factor `0.2 * sqrt(max(output, input))`. Otherwise use Keller Jordan's
            factor `sqrt(max(1, output / input))`.

    Returns:
        float | torch.Tensor: Shape adjusted learning rate.

    """
    output_shape, *input_shape = param_shape
    input_shape = math.prod(input_shape)

    ratio: float = (
        0.2 * math.sqrt(max(output_shape, input_shape))
        if use_adjusted_lr
        else math.pow(max(1.0, output_shape / input_shape), 0.5)
    )

    return lr * ratio


class MuonBase(BaseOptimizer):
    """Shared updates for Muon optimizers."""

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
                shapes: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

                for index, p in enumerate(tensors['params']):
                    rows, columns = p.size(0), p.numel() // p.size(0)
                    shape = (min(rows, columns), max(rows, columns)) if group['use_adjusted_lr'] else (rows, columns)
                    shapes[shape].append(index)

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
            torch._foreach_div_(de_noms, bias_correction2**0.5)
            torch._foreach_add_(de_noms, group['eps'])

            foreach_addcdiv_(params, exp_avgs, de_noms, -group['lr'] / bias_correction1)

    @staticmethod
    def _orthogonalize(group: ParamGroup, updates: list[torch.Tensor]) -> list[torch.Tensor]:
        matrices = [update.reshape(update.size(0), -1) for update in updates]
        if len(matrices) > 1 and any(has_dtensor and isinstance(matrix, DTensor) for matrix in matrices):
            return [MuonBase._orthogonalize(group, [matrix])[0] for matrix in matrices]

        transposed = [matrix.size(0) > matrix.size(1) for matrix in matrices]
        matrix = matrices[0] if len(matrices) == 1 else torch.stack([
            matrix.mT if transpose else matrix for matrix, transpose in zip(matrices, transposed)
        ])

        if group.get('ns_method') == 'gram_newton_schulz':
            result = zero_power_via_gram_newton_schulz(
                matrix, num_steps=group['ns_steps'], weights=group['ns_coeffs'], reset_steps=group['ns_reset_steps']
            )
        else:
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
