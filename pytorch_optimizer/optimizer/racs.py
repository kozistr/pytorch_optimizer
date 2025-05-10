import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


class RACS(BaseOptimizer):
    r"""Row and Column Scaled SGD.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. momentum factor.
    :param alpha: float. scaler.
    :param gamma: float. limiter threshold.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.05,
        gamma: float = 1.01,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RACS'

    def init_group(self, group: GROUP, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                state = self.state[p]

                if grad.ndim < 2:
                    grad = grad.reshape(len(grad), 1)
                elif grad.ndim > 2:
                    grad = grad.reshape(len(grad), -1)

                if len(state) == 0:
                    state['s'] = torch.zeros(grad.size(0), dtype=grad.dtype, device=grad.device)
                    state['q'] = torch.ones(grad.size(1), dtype=grad.dtype, device=grad.device)
                    state['theta'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                s, q = state['s'], state['q']

                grad_p2 = grad.pow(2)
                s.mul_(beta).add_(grad_p2.mean(dim=1), alpha=1.0 - beta)
                q.mul_(beta).add_(grad_p2.mean(dim=0), alpha=1.0 - beta)

                s_sq = s.add(group['eps']).sqrt_().unsqueeze(1)
                q_sq = q.add(group['eps']).sqrt_().unsqueeze(0)

                grad_hat = grad / (s_sq * q_sq)

                grad_hat_norm = torch.norm(grad_hat)
                threshold = (
                    group['gamma'] / max(grad_hat_norm / (state['theta'] + group['eps']), group['gamma'])
                    if group['step'] > 1
                    else 1.0
                )
                state['theta'] = grad_hat_norm.mul_(threshold)

                p.add_(grad_hat.view_as(p), alpha=-group['lr'] * group['alpha'] * threshold)

        return loss


class Alice(BaseOptimizer):
    r"""Adaptive low-dimensional subspace estimation.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
        beta3=0 for Alice-0 optimizer.
    :param alpha: float. scaler.
    :param alpha_c: float. compensation scaler.
    :param update_interval: int. update interval.
    :param rank: int. rank.
    :param gamma: limiter threshold.
    :param leading_basis: int. leading basis.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 0.02,
        betas: BETAS = (0.9, 0.9, 0.999),
        alpha: float = 0.3,
        alpha_c: float = 0.4,
        update_interval: int = 200,
        rank: int = 256,
        gamma: float = 1.01,
        leading_basis: int = 40,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_range(alpha_c, 'alpha_c', 0.0, 1.0)
        self.validate_positive(update_interval, 'update_interval')
        self.validate_positive(rank, 'rank')
        self.validate_positive(gamma, 'gamma')
        self.validate_positive(leading_basis, 'leading_basis')
        self.validate_non_negative(rank - leading_basis, 'rank - leading_basis')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'alpha_c': alpha_c,
            'update_interval': update_interval,
            'rank': rank,
            'gamma': gamma,
            'leading_basis': leading_basis,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Alice'

    def init_group(self, group: GROUP, **kwargs) -> None:
        pass

    @staticmethod
    def subspace_iteration(
        a: torch.Tensor, mat: torch.Tensor, num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Perform subspace iteration."""
        u = mat
        for _ in range(num_steps):
            u, _ = torch.linalg.qr(a @ u)

        return torch.linalg.eigh(u.T @ a @ u)

    def switch(self, q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
        vals, vecs = self.subspace_iteration(q.to(torch.float32), u_prev.to(torch.float32), num_steps=1)

        leading_indices = torch.argsort(vals, descending=True)[:leading_basis]
        u_t1 = vecs[:, leading_indices]

        u_c, _ = torch.linalg.qr(torch.eye(q.shape[0], device=q.device) - u_t1 @ u_t1.T)
        u_t2 = u_c[:, :rank - leading_basis]  # fmt: skip

        return torch.cat([u_t1, u_t2], dim=1).to(q.dtype)

    @staticmethod
    def compensation(
        grad: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        gamma: float,
        decay_rate: float,
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, n = grad.shape

        sigma = u.T @ grad

        p.mul_(decay_rate).add_(grad.pow(2).sum(dim=0) - sigma.pow(2).sum(dim=0), alpha=1.0 - decay_rate).clamp_min_(
            1e-8
        )

        d = torch.zeros_like(grad)
        diag_len: int = min(m, n)
        d[torch.arange(diag_len), torch.arange(diag_len)] = 1.0 / p.sqrt()[:diag_len]

        c_t = math.sqrt(m - rank) * (grad - u @ sigma) * d if m >= rank else torch.zeros_like(grad)

        n = gamma / max(torch.norm(c_t) / phi, gamma) if phi.item() > 0 else torch.ones_like(phi)

        c_t.mul_(n)
        phi = torch.norm(c_t)

        return c_t, phi

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2, beta3 = group['betas']
            rank, leading_basis = group['rank'], group['leading_basis']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                state = self.state[p]

                if grad.ndim < 2:
                    grad = grad.reshape(len(grad), 1)
                elif grad.ndim > 2:
                    grad = grad.reshape(len(grad), -1)

                if len(state) == 0:
                    m, n = grad.shape

                    state['U'] = torch.zeros((m, rank), dtype=p.dtype, device=p.device)
                    state['Q'] = torch.zeros((rank, rank), dtype=p.dtype, device=p.device)

                    state['m'] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)
                    state['v'] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)

                    state['p'] = torch.zeros((n,), dtype=p.dtype, device=p.device)
                    state['phi'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                q, u, m, v = state['Q'], state['U'], state['m'], state['v']

                if group['step'] == 1 or group['step'] % group['update_interval'] == 0:
                    q_t = beta3 * (u @ q @ u.T) + (1.0 - beta3) * (grad @ grad.T)
                    u = self.switch(q_t, u, rank, leading_basis)

                sigma = u.T @ grad

                q.mul_(beta3).add_(sigma @ sigma.T, alpha=1.0 - beta3)
                m.mul_(beta1).add_(sigma, alpha=1.0 - beta1)
                v.mul_(beta2).add_(sigma.pow(2), alpha=1.0 - beta2)

                c_t, phi = self.compensation(grad, u, state['p'], state['phi'], group['gamma'], beta1, rank)

                update = u @ (m / v.sqrt())
                update.add_(c_t, alpha=group['alpha_c'])

                p.add_(update.view_as(p), alpha=-group['lr'] * group['alpha'])

                state['phi'] = phi

        return loss
