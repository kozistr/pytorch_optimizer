import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class RACS(BaseOptimizer):
    """Row and Column Scaled SGD.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        beta: Decay rate for row- and column wise squared gradient averages.
        alpha: Update scaling factor.
        gamma: Maximum multiplicative growth of the scaled update norm.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates. `None` enables batching when supported. Defaults to `False`.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.05,
        gamma: float = 1.01,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RACS'

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

            if 's' not in state:
                if grad.ndim < 2:
                    grad = grad.reshape(grad.numel(), 1)
                elif grad.ndim > 2:
                    grad = grad.reshape(len(grad), -1)

                state['s'] = torch.zeros(grad.size(0), dtype=grad.dtype, device=grad.device)
                state['q'] = torch.ones(grad.size(1), dtype=grad.dtype, device=grad.device)
                state['theta'] = torch.zeros((), dtype=grad.dtype, device=grad.device)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        initialized: list[bool],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        grads = [grad.reshape(grad.size(0) if grad.ndim else 1, -1) for grad in grads]

        squares = torch._foreach_pow(grads, 2)
        torch._foreach_lerp_(state['s'], [value.mean(dim=1) for value in squares], weight=1.0 - group['beta'])
        torch._foreach_lerp_(state['q'], [value.mean(dim=0) for value in squares], weight=1.0 - group['beta'])

        for p, grad, s, q, theta, has_state in zip(params, grads, state['s'], state['q'], state['theta'], initialized):
            s_sq = s.add(group['eps']).sqrt_().unsqueeze(1)
            q_sq = q.add(group['eps']).sqrt_().unsqueeze(0)

            grad_hat = grad / (s_sq * q_sq)
            grad_hat_norm = torch.norm(grad_hat)
            threshold = (
                group['gamma'] / (grad_hat_norm / (theta + group['eps'])).clamp_min(group['gamma'])
                if has_state
                else 1.0
            )

            theta.copy_(grad_hat_norm * threshold)
            foreach_add_([p], [grad_hat.view_as(p)], alpha=-group['lr'] * group['alpha'] * threshold)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            new_params = {p for p in group['params'] if p.grad is not None and 's' not in self.state.get(p, {})}

            self.init_group(group)
            group['step'] += 1

            beta = group['beta']

            if self.can_use_foreach(group, group.get('foreach')):
                params, grads, state = self.collect_trainable_params(group, self.state, ['s', 'q', 'theta'])

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group,
                        batch['params'],
                        batch['grads'],
                        batch,
                        [p not in new_params for p in batch['params']],
                    )

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                self.maximize_gradient(grad, maximize=self.maximize)

                has_state = p not in new_params

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if grad.ndim < 2:
                    grad = grad.reshape(grad.numel(), 1)
                elif grad.ndim > 2:
                    grad = grad.reshape(len(grad), -1)

                s, q = state['s'], state['q']

                grad_p2 = grad.pow(2)
                s.lerp_(grad_p2.mean(dim=1), weight=1.0 - beta)
                q.lerp_(grad_p2.mean(dim=0), weight=1.0 - beta)

                s_sq = s.add(group['eps']).sqrt_().unsqueeze(1)
                q_sq = q.add(group['eps']).sqrt_().unsqueeze(0)

                grad_hat = grad_p2
                torch.mul(s_sq, q_sq, out=grad_hat)
                torch.div(grad, grad_hat, out=grad_hat)

                grad_hat_norm = torch.norm(grad_hat)
                threshold = (
                    group['gamma'] / max(grad_hat_norm / (state['theta'] + group['eps']), group['gamma'])
                    if has_state
                    else 1.0
                )
                state['theta'] = grad_hat_norm.mul_(threshold)

                p.add_(grad_hat.view_as(p), alpha=-group['lr'] * group['alpha'] * threshold)

        return loss


class Alice(BaseOptimizer):
    """Adaptive subspace updates with full rank gradient compensation.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for gradient momentum, squared gradients, and subspace statistics. Set the third value
            to 0 for Alice-0.
        alpha: Update scaling factor.
        alpha_c: Scaling factor for the compensation update.
        update_interval: Number of steps between subspace updates.
        rank: Dimension of the low rank subspace.
        gamma: Maximum multiplicative growth of the scaled update norm.
        leading_basis: Number of leading subspace basis vectors to update.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        betas: Betas = (0.9, 0.9, 0.999),
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

        defaults: Defaults = {
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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @staticmethod
    def subspace_iteration(
        a: torch.Tensor, mat: torch.Tensor, num_steps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Perform subspace iteration."""
        u = mat
        for _ in range(num_steps):
            u, _ = torch.linalg.qr(a @ u)

        vals, vecs = torch.linalg.eigh(u.T @ a @ u)
        return vals, u @ vecs

    def switch(self, q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
        vals, vecs = self.subspace_iteration(q.to(torch.float32), u_prev.to(torch.float32), num_steps=1)

        leading_indices = torch.argsort(vals, descending=True)[:leading_basis]
        u_t1 = vecs[:, leading_indices]

        if rank == leading_basis:
            return u_t1.to(q.dtype)

        u_c, _ = torch.linalg.qr(vecs, mode='complete')
        u_c = u_c[:, rank:]

        if u_c.size(1) < rank - leading_basis:
            remaining_indices = torch.argsort(vals, descending=True)[leading_basis:]
            u_c = torch.cat([u_c, vecs[:, remaining_indices]], dim=1)

        indices = torch.randperm(u_c.size(1), device=q.device)[:rank - leading_basis]
        u_t2 = u_c[:, indices]

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = grad.size(0)

        sigma = u.T @ grad

        p.lerp_(grad.pow(2).sum(dim=0) - sigma.pow(2).sum(dim=0), weight=1.0 - decay_rate).clamp_min_(1e-8)

        c_t = math.sqrt(m - rank) * (grad - u @ sigma) / p.sqrt() if m >= rank else torch.zeros_like(grad)

        n = gamma / max(torch.norm(c_t) / phi, gamma) if phi.item() > 0 else torch.ones_like(phi)

        c_t.mul_(n)
        phi = torch.norm(c_t)

        return c_t, phi

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                state = self.state[p]

                self.maximize_gradient(grad, maximize=self.maximize)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if grad.ndim < 2:
                    grad = grad.reshape(grad.numel(), 1)
                elif grad.ndim > 2:
                    grad = grad.reshape(len(grad), -1)

                has_state = 'U' in state
                if not has_state:
                    m, n = grad.shape
                    rank = min(group['rank'], m)

                    state['U'] = torch.zeros((m, rank), dtype=p.dtype, device=p.device)
                    state['Q'] = torch.zeros((rank, rank), dtype=p.dtype, device=p.device)

                    state['m'] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)
                    state['v'] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)

                    state['p'] = torch.zeros((n,), dtype=p.dtype, device=p.device)
                    state['phi'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                rank = state['U'].size(1)
                leading_basis = min(group['leading_basis'], rank)

                q, u, m, v = state['Q'], state['U'], state['m'], state['v']

                if not has_state or group['step'] % group['update_interval'] == 0:
                    q_t = beta3 * (u @ q @ u.T) + (1.0 - beta3) * (grad @ grad.T)
                    u = self.switch(q_t, u, rank, leading_basis)
                    state['U'] = u

                sigma = u.T @ grad

                q.lerp_(sigma @ sigma.T, weight=1.0 - beta3)
                m.lerp_(sigma, weight=1.0 - beta1)
                v.lerp_(sigma.pow(2), weight=1.0 - beta2)

                c_t, phi = self.compensation(grad, u, state['p'], state['phi'], group['gamma'], beta1, rank)

                update = u @ (m / v.sqrt().add_(group['eps']))
                update.add_(c_t, alpha=group['alpha_c'])

                p.add_(update.view_as(p), alpha=-group['lr'] * group['alpha'])

                state['phi'] = phi

        return loss
