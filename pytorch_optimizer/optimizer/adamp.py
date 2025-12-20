import math
from typing import Callable, List, Tuple

import torch
from torch.nn.functional import cosine_similarity

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient


def channel_view(x: torch.Tensor) -> torch.Tensor:
    """Do channel view."""
    return x.view(x.size()[0], -1)


def layer_view(x: torch.Tensor) -> torch.Tensor:
    """Do layer view."""
    return x.view(1, -1)


def cosine_similarity_by_view(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    view_func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Calculate cosine similarity by the view.

    Args:
        x (torch.Tensor): Source tensor.
        y (torch.Tensor): Destination tensor.
        eps (float): Small constant epsilon added for numerical stability.
        view_func (Callable): Function defining the view (e.g., per-channel or per-layer).
    """
    x = view_func(x)
    y = view_func(y)
    return cosine_similarity(x, y, dim=1, eps=eps).abs_()


def projection(
    p: torch.Tensor,
    grad: torch.Tensor,
    perturb: torch.Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> Tuple[torch.Tensor, float]:
    """Project to remove the radial component from the update vector."""
    wd: float = 1.0
    expand_size: List[int] = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (channel_view, layer_view):
        cosine_sim = cosine_similarity_by_view(grad, p, eps, view_func)

        if cosine_sim.max() < delta / math.sqrt(view_func(p).size()[1]):
            p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class SGDP(BaseOptimizer):
    """SGD + Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        dampening (float): Dampening for momentum.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        delta (float): Threshold that determines whether a set of parameters is scale-invariant or not.
        wd_ratio (float): Relative weight decay applied on scale-invariant parameters compared to that applied
            on scale-variant parameters.
        nesterov (bool): Enables Nesterov momentum.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'momentum': momentum,
            'dampening': dampening,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDP'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]
            if len(state) == 0:
                state['momentum'] = torch.zeros_like(grad)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                buf = state['momentum']

                p, grad, buf = self.view_as_real(p, grad, buf)

                buf.mul_(momentum).add_(grad, alpha=1.0 - group['dampening'])

                d_p = buf.clone()
                if group['nesterov']:
                    d_p = d_p.mul_(momentum).add_(grad)

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = projection(
                        p,
                        grad,
                        d_p,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=wd_ratio / (1.0 - momentum),
                )

                p.add_(d_p, alpha=-group['lr'])

        return loss


class AdamP(BaseOptimizer):
    """Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        delta (float): Threshold that determines whether a set of parameters is scale-invariant or not.
        wd_ratio (float): Relative weight decay applied on scale-invariant parameters compared to that applied
            on scale-variant parameters.
        nesterov (bool): Enables Nesterov momentum.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamP'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

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

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

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

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'],
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                if group.get('use_gc'):
                    centralize_gradient(grad, gc_conv_only=False)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                inv_de_nom = exp_avg_sq.rsqrt().add_(group['eps']).mul_(bias_correction2_sq)

                perturb = exp_avg.clone()

                if group.get('cautious'):
                    self.apply_cautious(perturb, grad)

                if group['nesterov']:
                    perturb.mul_(beta1).addcmul_(grad, inv_de_nom, value=1.0 - beta1)
                else:
                    perturb.mul_(inv_de_nom)

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=wd_ratio,
                )

                p.add_(perturb, alpha=-step_size)

        return loss
