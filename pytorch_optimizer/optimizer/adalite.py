import torch
from torch.nn.functional import softmax

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class Adalite(BaseOptimizer):
    r"""Adalite optimizer.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas: Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        g_norm_min (float): Minimum gradient norm threshold.
        ratio_min (float): Minimum ratio value for adaptive adjustment.
        tau (float): Time constant controlling parameter smoothing or decay behavior.
        eps1 (float): Term added to the denominator to improve numerical stability.
        eps2 (float): Additional term added to the denominator for extra numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-2,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        g_norm_min: float = 1e-10,
        ratio_min: float = 1e-4,
        tau: float = 1.0,
        eps1: float = 1e-6,
        eps2: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'g_norm_min': g_norm_min,
            'ratio_min': ratio_min,
            'tau': tau,
            'eps1': eps1,
            'eps2': eps2,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adalite'

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
                if len(p.shape) < 2:
                    state['m_avg'] = torch.zeros_like(p)
                    state['v_avg'] = torch.zeros_like(p)
                else:
                    state['v_avg_0'] = torch.zeros_like(p.mean(dim=1))
                    state['v_avg_1'] = torch.zeros_like(p.mean(dim=0))

                    state['m_avg_c'] = torch.zeros_like(p.mean(dim=1)[:, None])
                    state['m_avg_r'] = torch.zeros_like(p.mean(dim=0)[None, :])
                    state['m_avg_u'] = torch.zeros_like(p.mean().unsqueeze(0).unsqueeze(0))

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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if sum(grad.shape) > 1:
                    trust_ratio = (p.norm() / grad.norm().clip(min=group['g_norm_min'])).clip(min=group['ratio_min'])
                    grad.mul_(trust_ratio)

                if len(grad.shape) < 2:
                    m = state['m_avg']
                    v = state['v_avg']
                else:
                    r, c = state['v_avg_0'][:, None], state['v_avg_1'][None, :]
                    v = (r * c) / r.sum().clamp(min=group['eps2'])
                    m = state['m_avg_c'] @ state['m_avg_u'] @ state['m_avg_r']

                m.lerp_(grad, 1.0 - beta1)
                v.lerp_((grad - m).square(), 1.0 - beta2)

                v_avg = v / (1.0 - beta2 ** group['step'])

                if len(grad.shape) == 2:
                    imp_c = softmax(v.mean(dim=1), dim=0)[:, None]
                    imp_r = softmax(v.mean(dim=0), dim=0)[None, :]
                    m.lerp_(grad, 1.0 - imp_c * imp_r)

                u = m.lerp(grad, 1.0 - beta1)

                if len(grad.shape) < 2:
                    state['m_avg'] = m
                    state['v_avg'] = v
                else:
                    state['v_avg_0'] = v.sum(dim=1)
                    state['v_avg_1'] = v.sum(dim=0) / v.sum().clamp(min=group['eps2'])

                    imp_c = softmax(v.mean(dim=1) / group['tau'], dim=-1)[:, None]
                    imp_r = softmax(v.mean(dim=0) / group['tau'], dim=-1)[None, :]

                    c = ((m * imp_r).sum(dim=1))[:, None]
                    r = ((m * imp_c).sum(dim=0))[None, :]

                    s = (c.T @ m @ r.T) / (c.T @ c @ r @ r.T).clamp(min=group['eps2'])

                    state['m_avg_c'] = c
                    state['m_avg_r'] = r
                    state['m_avg_u'] = s

                u.div_((v_avg + group['eps1']).sqrt())

                u = u.reshape(p.shape)
                u.add_(p, alpha=group['weight_decay'])

                p.add_(u, alpha=-group['lr'])

        return loss
