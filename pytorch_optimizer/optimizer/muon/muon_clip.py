from typing import Literal

import torch

from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon.muon import Muon
from pytorch_optimizer.optimizer.utils.matrix import DTensor, NewtonSchulzWeights, has_dtensor


class MuonClip(Muon):
    """Muon with per-head query/key weight clipping from the Kimi K2 report.

    Use Muon's `use_muon` parameter groups. Label attention projection groups with
    `attention_name` and `qk_clip`: `query`, `key`, `query_rotary`, or `key_rotary`.
    Rows must contain contiguous head blocks, or the first axis must index heads.
    Pass maximum pre-softmax logits per head to `step(attention_max_logits=...)`.
    For MLA, use separate content and rotary groups with `qk_clip_alpha=0.5`.
    Shared rotary keys are left unchanged by clipping.

    Reference:
        https://arxiv.org/abs/2507.20534

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate for Muon groups.
        momentum: Momentum factor.
        weight_decay: Decoupled weight decay coefficient for Muon groups.
        qk_clip_threshold: Maximum attention logit threshold.
        qk_clip_alpha: Query scaling exponent. Keys use `1 - qk_clip_alpha`.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        ns_method: Orthogonalization method, `newton_schulz` or `gram_newton_schulz`.
        ns_reset_steps: Iteration counts after which to restart Gram Newton-Schulz.
        use_adjusted_lr: Use the Moonlight shape adjustment, enabled by default.
        adamw_lr: Learning rate for AdamW groups.
        adamw_betas: Decay rates for the AdamW moments.
        adamw_wd: Decoupled weight decay for AdamW groups.
        adamw_eps: Numerical stability constant for AdamW groups.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes.
        **kwargs (dict): Additional parameter group defaults.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        ns_method: Literal['newton_schulz', 'gram_newton_schulz'] = 'newton_schulz',
        ns_reset_steps: tuple[int, ...] = (2,),
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = None,
        use_adjusted_lr: bool = True,
        **kwargs,
    ):
        self.validate_positive(qk_clip_threshold, 'qk_clip_threshold')
        self.validate_range(qk_clip_alpha, 'qk_clip_alpha', 0.0, 1.0, range_type='[]')

        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=False,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
            ns_method=ns_method,
            ns_reset_steps=ns_reset_steps,
            use_adjusted_lr=use_adjusted_lr,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
            maximize=maximize,
            foreach=foreach,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            **kwargs,
        )

        for group in self.param_groups:
            self.validate_positive(group['qk_clip_threshold'], 'qk_clip_threshold')
            self.validate_range(group['qk_clip_alpha'], 'qk_clip_alpha', 0.0, 1.0, range_type='[]')
            if group.get('qk_clip') is not None:
                self.validate_options(group['qk_clip'], 'qk_clip', ['query', 'key', 'query_rotary', 'key_rotary'])
                if 'attention_name' not in group:
                    raise ValueError('`attention_name` must be set for QK-Clip groups.')

    def __str__(self) -> str:
        return 'MuonClip'

    def _momentum_updates(
        self, group: ParamGroup, grads: list[torch.Tensor], buffers: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        torch._foreach_mul_(buffers, group['momentum'])
        torch._foreach_add_(buffers, grads)
        return self._orthogonalize(group, buffers)

    @torch.no_grad()
    def step(self, closure: Closure = None, attention_max_logits: dict[str, torch.Tensor] | None = None) -> Loss:
        """Update parameters, then clip labeled projections using this batch's per-head maximum logits.

        Args:
            closure: Callable that reevaluates the model and returns the loss.
            attention_max_logits: Mapping from attention names to one-dimensional per-head maximum logits.
                Reduce over all training microbatches and distributed ranks before passing these values.
                Omitted attention names are not clipped.

        Returns:
            Loss: Loss returned by the closure, or `None`.

        """
        loss = super().step(closure)
        if attention_max_logits is None:
            return loss

        for group in self.param_groups:
            role = group.get('qk_clip')
            if role is None or role == 'key_rotary' or group['attention_name'] not in attention_max_logits:
                continue

            logits = attention_max_logits[group['attention_name']].detach().float()
            if logits.ndim != 1 or logits.numel() == 0:
                raise ValueError('attention maximum logits must be a nonempty one-dimensional tensor.')

            alpha = group['qk_clip_alpha']
            exponent = {'query': alpha, 'key': 1.0 - alpha, 'query_rotary': 1.0}[role]
            scales = (group['qk_clip_threshold'] / logits.clamp_min(group['qk_clip_threshold'])).pow_(exponent)
            for p in group['params']:
                if p.ndim == 0 or p.size(0) % logits.numel() != 0:
                    raise ValueError('the first parameter dimension must be divisible by the attention head count.')

                row_scales = scales.to(device=p.device, dtype=p.dtype).repeat_interleave(p.size(0) // logits.numel())
                row_scales = row_scales.view(-1, *([1] * (p.ndim - 1)))
                if has_dtensor and isinstance(p, DTensor):
                    row_scales = DTensor.from_local(row_scales, device_mesh=p.device_mesh)
                    placements = [
                        placement if placement.is_shard() and placement.dim == 0 else replicated
                        for placement, replicated in zip(p.placements, row_scales.placements)
                    ]
                    row_scales = row_scales.redistribute(placements=placements)
                    p.to_local().mul_(row_scales.to_local())
                else:
                    p.mul_(row_scales)

        return loss
