import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class Aida(BaseOptimizer):
    """A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        k (int): Number of vectors projected per iteration.
        xi (float): Term used in vector projections to avoid division by zero.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        rectify (bool): Perform the rectified update similar to RAdam.
        n_sma_threshold (int): Number of SMA threshold (recommended is 5).
        degenerated_to_sgd (bool): Perform SGD update when variance of gradient is high.
        ams_bound (bool): Whether to use the AMSBound variant.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        k: int = 2,
        xi: float = 1e-20,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        rectify: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(k, 'k')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(xi, 'xi')
        self.validate_non_negative(eps, 'eps')

        self.k = k
        self.xi = xi
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'ams_bound': ams_bound,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Aida'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
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
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)

                if group['ams_bound']:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

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

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group['rectify'],
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)

                proj_g = grad.detach().clone()
                proj_m = exp_avg.detach().clone()

                for _ in range(self.k):
                    proj_sum_gm = torch.sum(torch.mul(proj_g, proj_m))

                    scalar_g = proj_sum_gm / (torch.sum(torch.pow(proj_g, 2)).add_(self.xi))
                    scalar_m = proj_sum_gm / (torch.sum(torch.pow(proj_m, 2)).add_(self.xi))

                    proj_g.mul_(scalar_g)
                    proj_m.mul_(scalar_m)

                grad_residual = proj_m - proj_g
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2).add_(group['eps'])

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_var,
                    max_exp_avg_sq=state.get('max_exp_avg_var', None),
                    eps=group['eps'],
                )

                if not group['rectify']:
                    de_nom.div_(bias_correction2_sq)
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                    continue

                if n_sma >= self.n_sma_threshold:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                elif step_size > 0:
                    p.add_(exp_avg, alpha=-step_size)

        return loss
