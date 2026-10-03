import math
import random

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT, State


class SaRA(BaseOptimizer):
    """AdamW updates on progressively refined masks of small magnitude weights.

    Implements the parameter based reference in `sjtuplayer/SaRA/optim/adamw2.py`. Only weights with initial absolute
    values below `threshold` receive AdamW updates, including weight decay. Moments are stored only for these weights.
    Mask refinement keeps their accumulated moments. This optimizer uses ordinary PyTorch backpropagation rather than
    the paper's model reparameterization for unstructured backpropagation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate. Defaults to `1e-3 * exp(-350 * threshold)` when None.
        betas: Coefficients used for computing running averages of gradient and its square.
        threshold: Strict upper bound on the absolute values of initially trainable weights.
        progressive_iter: Refine the mask before update `progressive_iter + 1`. -1 disables refinement.
        lambda_rank: Nuclear norm penalty coefficient. Each step samples one matrix per parameter group with both
            dimensions greater than 64 and applies the penalty if more than 100 weights are selected.
        weight_decay: Weight decay coefficient.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | None = None,
        betas: Betas = (0.9, 0.999),
        threshold: float = 1e-3,
        progressive_iter: int = -1,
        lambda_rank: float = 0.0,
        weight_decay: float = 1e-2,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(threshold, 'threshold')
        self.validate_boundary(progressive_iter, -1, bound_type='lower')
        self.validate_non_negative(lambda_rank, 'lambda_rank')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: Defaults = {
            'lr': 1e-3 * math.exp(-350.0 * threshold) if lr is None else lr,
            'betas': betas,
            'threshold': threshold,
            'progressive_iter': progressive_iter,
            'lambda_rank': lambda_rank,
            'weight_decay': weight_decay,
            'ams_bound': ams_bound,
            'eps': eps,
            'maximize': maximize,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SaRA'

    def add_param_group(self, param_group: ParamGroup) -> None:
        super().add_param_group(param_group)

        group = self.param_groups[-1]
        for p in group['params']:
            self.state[p]['mask'] = p.detach().abs() < group['threshold']

    def load_state_dict(self, state_dict: State) -> None:
        super().load_state_dict(state_dict)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = self.state[p]['mask'].bool()

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p[state['mask']])
                state['exp_avg_sq'] = torch.zeros_like(state['exp_avg'])

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(state['exp_avg'])

    @torch.no_grad()
    def update_mask(self, group: ParamGroup) -> None:
        """Keep only initially selected weights that are still below the threshold and retain their moments."""
        for p in group['params']:
            state = self.state[p]

            mask = state['mask']
            keep = p[mask].abs() < group['threshold']

            state['mask'] = mask.clone()
            state['mask'][mask] = keep

            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    state[key] = state[key][keep]

    @staticmethod
    @torch.no_grad()
    def apply_rank_constraint(
        p: torch.Tensor, grad_mask: torch.Tensor, mask: torch.Tensor, lambda_rank: float
    ) -> None:
        """Add the masked nuclear norm subgradient to the selected gradient entries."""
        dtype = torch.float32 if p.dtype in (torch.float16, torch.bfloat16) else p.dtype

        matrix = torch.where(mask, p, 0.0).to(dtype=dtype)

        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        torch.mm(u, vh, out=matrix)

        grad_mask.add_(matrix[mask].to(dtype=grad_mask.dtype), alpha=lambda_rank)

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
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] / bias_correction1

            if group['progressive_iter'] >= 0 and group['step'] == group['progressive_iter'] + 1:
                self.update_mask(group)

            rank_params = [
                p
                for p in group['params']
                if group['lambda_rank'] > 0.0 and p.grad is not None and p.dim() == 2 and min(p.shape) > 64
            ]
            rank_param = random.choice(rank_params) if rank_params else None  # noqa: S311

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                mask = state['mask']

                grad_mask = p.grad[mask]

                if p is rank_param and mask.sum() > 100:
                    self.apply_rank_constraint(p, grad_mask, mask, group['lambda_rank'])

                self.maximize_gradient(grad_mask, maximize=group['maximize'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.lerp_(grad_mask, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_mask, grad_mask, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq'),
                    eps=0.0,
                    exp_avg_sq_eps=0.0,
                )
                de_nom.div_(bias_correction2_sq).add_(group['eps'])

                p_mask = p[mask]

                self.apply_weight_decay(
                    p=p_mask,
                    grad=grad_mask,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                p_mask.addcdiv_(exp_avg, de_nom, value=-step_size)
                p[mask] = p_mask

        return loss
