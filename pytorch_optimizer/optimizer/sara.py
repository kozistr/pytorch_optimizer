from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class SaRA(BaseOptimizer):
    r"""Progressive sparse low-rank adaptation.

    Entries whose absolute value is below ``threshold`` are updated with AdamW. The remaining entries are copied
    back after the step, so they do not move and do not keep the weight-decay factor. Moments are still updated
    from the full gradient, matching ``optim/adamw2.py`` in the reference implementation. When
    ``progressive_iter`` is non-negative, the step ``progressive_iter + 1`` replaces the mask with its intersection
    against the entries that are still below ``threshold``. A negative ``progressive_iter`` keeps the mask built at
    construction. ``lambda_rank`` adds the nuclear norm of one randomly chosen large matrix, restricted to the
    mask, before that AdamW step.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Decoupled weight decay.
        threshold (float): Entries with absolute value strictly below this are trainable.
        progressive_iter (int): Step after which the mask is intersected with the current small-magnitude set.
            Negative disables that refresh.
        lambda_rank (float): Weight of the nuclear-norm penalty. Zero disables it.
        amsgrad (bool): Whether to use the AMSGrad variant.
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.
        foreach (Optional[bool]): Accepted for the common optimizer signature. The update is per parameter.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        threshold: float = 1e-3,
        progressive_iter: int = -1,
        lambda_rank: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(threshold, 'threshold')
        self.validate_non_negative(lambda_rank, 'lambda_rank')
        if not isinstance(progressive_iter, int) or progressive_iter < -1:
            raise ValueError('progressive_iter must be an integer greater than or equal to -1')

        self.maximize = maximize
        self.foreach = foreach
        self.inner_iter: int = 0

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'threshold': threshold,
            'progressive_iter': progressive_iter,
            'lambda_rank': lambda_rank,
            'amsgrad': amsgrad,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                mask = p.detach().abs().lt(group['threshold'])
                self.state[p]['initial_mask'] = mask
                self.state[p]['mask'] = mask.clone()

    def __str__(self) -> str:
        return 'SaRA'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise NoSparseGradientError(str(self))
            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if 'exp_avg' not in state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group.get('amsgrad', False):
                    state['max_exp_avg_sq'] = torch.zeros_like(p)
            if 'mask' not in state:
                mask = p.detach().abs().lt(group.get('threshold', 1e-3))
                state['initial_mask'] = mask
                state['mask'] = mask.clone()

    def _refresh_masks(self, group: ParamGroup) -> None:
        if group['progressive_iter'] < 0 or self.inner_iter != group['progressive_iter'] + 1:
            return
        for p in group['params']:
            state = self.state[p]
            fresh = p.detach().abs().lt(group['threshold'])
            state['mask'] = fresh.logical_and(state['initial_mask'])

    def _apply_rank_penalty(self, group: ParamGroup) -> None:
        if group['lambda_rank'] == 0.0:
            return
        candidates = [
            p for p in group['params'] if p.ndim == 2 and min(p.shape) > 64 and int(self.state[p]['mask'].sum()) > 100
        ]
        if not candidates:
            return
        chosen = candidates[int(torch.randint(len(candidates), ()).item())]
        mask = self.state[chosen]['mask']
        with torch.enable_grad():
            penalty = group['lambda_rank'] * torch.linalg.svdvals(chosen * mask).sum()
            penalty.backward()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        self.inner_iter += 1
        for group in self.param_groups:
            self._refresh_masks(group)
            self._apply_rank_penalty(group)

        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = -p.grad if self.maximize else p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                original = p.detach().clone()

                p.mul_(1.0 - group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state['step'] += 1
                step = state['step']
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = group['lr'] / bias_correction1
                if group['amsgrad']:
                    torch.maximum(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                    denom = state['max_exp_avg_sq'].sqrt().div(bias_correction2**0.5).add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().div(bias_correction2**0.5).add(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                kept = state['mask'].to(dtype=p.dtype)
                p.mul_(kept).add_(original.mul(1.0 - kept))

        return loss
