import math
from typing import Literal, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5

MARS_TYPE = Literal['adamw', 'lion', 'shampoo']


class MARS(BaseOptimizer):
    """Unleashing the Power of Variance Reduction for Training Large Models.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        gamma (float): The scaling parameter that controls the strength of gradient correction.
        mars_type (MARS_TYPE): Type of MARS. Supported types are `adamw`, `lion`, `shampoo`.
        optimize_1d (bool): Whether MARS should optimize 1D parameters.
        lr_1d (float): Learning rate for AdamW when optimize_1d is set to False.
        betas_1d (Betas): Coefficients for running averages of gradient and squared Hessian for 1D.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decay_1d (float): Weight decay for 1D parameters.
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        ams_bound (bool): Whether to use the AMSBound variant.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 3e-3,
        betas: Betas = (0.95, 0.99),
        gamma: float = 0.025,
        mars_type: MARS_TYPE = 'adamw',
        optimize_1d: bool = False,
        lr_1d: bool = 3e-3,
        betas_1d: Betas = (0.9, 0.95),
        weight_decay: float = 0.0,
        weight_decay_1d: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        cautious: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(lr_1d)
        self.validate_betas(betas)
        self.validate_betas(betas_1d)
        self.validate_options(mars_type, 'mars_type', ['adamw', 'lion', 'shampoo'])
        self.validate_non_negative(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(weight_decay_1d, 'weight_decay_1d')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'lr_1d': lr_1d,
            'lr_1d_factor': lr_1d / lr,
            'betas': betas,
            'betas_1d': betas_1d,
            'mars_type': mars_type,
            'gamma': gamma,
            'optimize_1d': optimize_1d,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'cautious': cautious,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MARS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['last_grad'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    def optimize_mixed(
        self,
        grad: torch.Tensor,
        last_grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        betas: Tuple[int, int],
        gamma: float,
        mars_type: MARS_TYPE,
        is_grad_2d: bool,
        step: int,
        ams_bound: bool,
        cautious: bool,
        eps: float,
    ) -> torch.Tensor:
        beta1, beta2 = betas

        c_t = (grad - last_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)
        c_t_norm = torch.norm(c_t)
        if c_t_norm > 1.0:
            c_t.div_(c_t_norm)

        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)

        update = exp_avg.clone()
        if cautious:
            self.apply_cautious(update, grad)

        if mars_type == 'adamw' or (mars_type == 'shampoo' and not is_grad_2d):
            exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

            bias_correction1: float = self.debias(beta1, step)
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

            de_nom = self.apply_ams_bound(ams_bound, exp_avg_sq, max_exp_avg_sq, eps)
            de_nom.div_(bias_correction2_sq).mul_(bias_correction1)

            return update.div_(de_nom)

        if mars_type == 'lion':
            return update.sign_()

        factor: float = math.sqrt(max(1.0, grad.size(0) / grad.size(1)))

        update = update.view(update.size(0), -1)

        return zero_power_via_newton_schulz_5(update.mul_(1.0 / (1.0 - beta1)), eps=eps).mul_(factor).view_as(grad)

    def optimize_1d(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        betas: Tuple[int, int],
        step: int,
        ams_bound: bool,
        cautious: bool,
        eps: float,
    ) -> torch.Tensor:
        beta1, beta2 = betas

        bias_correction1: float = self.debias(beta1, step)
        bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        update = exp_avg.clone()

        if cautious:
            self.apply_cautious(update, grad)

        de_nom = self.apply_ams_bound(ams_bound, exp_avg_sq, max_exp_avg_sq, eps)
        de_nom.div_(bias_correction2_sq).mul_(bias_correction1)

        return update.div_(de_nom)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, last_grad = state['exp_avg'], state['exp_avg_sq'], state['last_grad']

                p, grad, exp_avg, exp_avg_sq, last_grad = self.view_as_real(p, grad, exp_avg, exp_avg_sq, last_grad)

                is_grad_2d: bool = grad.ndim >= 2
                step_size: float = (
                    group['lr'] if group['optimize_1d'] or is_grad_2d else group['lr'] * group['lr_1d_factor']
                )

                if group['optimize_1d'] or is_grad_2d:
                    update = self.optimize_mixed(
                        grad,
                        last_grad,
                        exp_avg,
                        exp_avg_sq,
                        state.get('max_exp_avg_sq', None),
                        group['betas'],
                        group['gamma'],
                        group['mars_type'],
                        is_grad_2d,
                        group['step'],
                        group['ams_bound'],
                        group.get('cautious'),
                        group['eps'],
                    )
                else:
                    update = self.optimize_1d(
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        state.get('max_exp_avg_sq', None),
                        group['betas_1d'],
                        group['step'],
                        group['ams_bound'],
                        group.get('cautious'),
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=step_size,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.add_(update, alpha=-step_size)

                state['last_grad'] = torch.view_as_complex(grad) if torch.is_complex(state['last_grad']) else grad

        return loss
