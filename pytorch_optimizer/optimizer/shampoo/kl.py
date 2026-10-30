from collections import defaultdict

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class KLShampoo(BaseOptimizer):
    """Shampoo with covariance and eigenvalue estimation through KL minimization.

    Reference: https://arxiv.org/abs/2509.03378.
    Scalars, vectors, and tensors exceeding `max_precondition_dim` use Adam updates without bias correction.
    The first gradient of each preconditioned parameter initializes its eigenbases without updating the parameter.
    Optimizer states use at least float32 precision.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for momentum and second moments. Bias correction is not applied.
        shampoo_beta: Decay rate for covariance and eigenvalue statistics. `None` uses `beta2`.
        weight_decay: Decoupled weight decay coefficient.
        precondition_frequency: Number of updates between eigenbasis refreshes, after the first nine updates.
        max_precondition_dim: Largest tensor dimension eligible for preconditioning.
        init_factor: Initial eigenvalue estimate. Must be positive.
        max_preconditioner_value: Upper bound on each factor's inverse square root eigenvalues.
            `None` disables clamping.
        eps: Damping added to the preconditioning denominator.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch compatible matrix and diagonal updates. `None` enables batching.

    """

    _use_soap: bool = False
    _supports_compiled_foreach: bool = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-4,
        betas: Betas = (0.9, 0.98),
        shampoo_beta: float | None = None,
        weight_decay: float = 1e-2,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 10000,
        init_factor: float = 1e-3,
        max_preconditioner_value: float | None = 4000.0,
        eps: float = 1e-9,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        if shampoo_beta is not None:
            self.validate_range(shampoo_beta, 'shampoo_beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(precondition_frequency, 'precondition_frequency')
        self.validate_positive(max_precondition_dim, 'max_precondition_dim')
        self.validate_positive(init_factor, 'init_factor')
        if max_preconditioner_value is not None:
            self.validate_positive(max_preconditioner_value, 'max_preconditioner_value')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'weight_decay': weight_decay,
            'precondition_frequency': precondition_frequency,
            'max_precondition_dim': max_precondition_dim,
            'init_factor': init_factor,
            'max_preconditioner_value': max_preconditioner_value,
            'eps': eps,
            'foreach': foreach,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return self.__class__.__name__

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compile_steps(
            self._step_foreach, self._step_diagonal, self._refresh_bases, compile_kwargs=compile_kwargs
        )

    def _restore_state_types(self, value, saved_value):
        if isinstance(saved_value, torch.Tensor) and saved_value.is_floating_point():
            return saved_value.to(device=value.device)

        return super()._restore_state_types(value, saved_value)

    @staticmethod
    def mode_product(grad: torch.Tensor, matrix: torch.Tensor, dim: int) -> torch.Tensor:
        """Multiply one tensor dimension by the columns of a matrix."""
        return torch.tensordot(grad, matrix, dims=([dim], [0])).movedim(-1, dim)

    @classmethod
    def project(cls, grad: torch.Tensor, bases: list[torch.Tensor], backward: bool = False) -> torch.Tensor:
        """Transform a tensor into or out of the Kronecker eigenbasis."""
        for dim, basis in enumerate(bases):
            grad = cls.mode_product(grad, basis.T if backward else basis, dim)
        return grad

    @staticmethod
    def inverse_root(
        inv_root: torch.Tensor, estimate: torch.Tensor, beta: float, max_value: float | None
    ) -> torch.Tensor:
        """Update an inverse square root estimate with optional clamping."""
        eigenvalues = inv_root.square().reciprocal().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues.lerp_(estimate, weight=1.0 - beta)
        result = eigenvalues.rsqrt_().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return result if max_value is None else result.clamp_(max=max_value)

    @staticmethod
    def qr_basis(matrix: torch.Tensor) -> torch.Tensor:
        """Return a QR basis with a positive diagonal in R."""
        q, r = torch.linalg.qr(matrix)
        signs = r.diagonal(dim1=-2, dim2=-1).sign()
        signs.masked_fill_(signs == 0, 1.0)
        return q * signs.unsqueeze(-2)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise NoSparseGradientError(str(self))
            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if len(state) != 0:
                continue

            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, dtype=dtype)
            state['preconditioned'] = p.ndim > 1 and max(p.shape) <= group['max_precondition_dim']
            if self._use_soap or not state['preconditioned']:
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=dtype)

            if state['preconditioned']:
                state['GG'], state['Q'], state['inv_roots'] = [], [], []
                grad = p.grad.to(dtype=dtype)
                for dim, size in enumerate(grad.shape):
                    unfolded = grad.movedim(dim, 0).reshape(size, -1)
                    statistic = unfolded @ unfolded.T
                    _, basis = torch.linalg.eigh(statistic)
                    statistic.div_(unfolded.shape[1]).diagonal().add_(1e-8)

                    state['GG'].append(statistic)
                    state['Q'].append(basis.flip(dims=(1,)))
                    state['inv_roots'].append(
                        torch.full((size,), group['init_factor'] ** -0.5, device=p.device, dtype=dtype)
                    )

    def update_preconditioner(self, grad: torch.Tensor, state: dict, group: ParamGroup) -> None:
        beta = group['shampoo_beta'] if group['shampoo_beta'] is not None else group['betas'][1]
        bases, inv_roots = state['Q'], state['inv_roots']
        projected = self.project(grad, bases)
        new_inv_roots = []

        for dim, size in enumerate(grad.shape):
            whitened, diagonal = grad, projected
            for other, (basis, inv_root) in enumerate(zip(bases, inv_roots)):
                if other == dim:
                    continue
                shape = [1] * grad.ndim
                shape[other] = -1
                whitened = self.mode_product(whitened, basis, other) * inv_root.reshape(shape)
                diagonal = diagonal * inv_root.reshape(shape)

            unfolded = whitened.movedim(dim, 0).reshape(size, -1)
            state['GG'][dim].mul_(beta).add_(unfolded @ unfolded.T, alpha=(1.0 - beta) / unfolded.shape[1])

            estimate = diagonal.movedim(dim, 0).reshape(size, -1).square().mean(dim=1)
            new_inv_roots.append(self.inverse_root(inv_roots[dim], estimate, beta, group['max_preconditioner_value']))

        state['inv_roots'] = new_inv_roots

        step = state['step'] - 1
        if step < 10 or step % group['precondition_frequency'] == 0:
            if self._use_soap:
                momentum = self.project(state['exp_avg'], bases, backward=True)
            for dim, (statistic, basis) in enumerate(zip(state['GG'], bases)):
                bases[dim] = self.qr_basis(statistic @ basis)
            if self._use_soap:
                state['exp_avg'].copy_(self.project(momentum, bases))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: dict[str, list[torch.Tensor]],
    ) -> None:
        if self.maximize:
            grads = [grad.to(dtype=momentum.dtype) for grad, momentum in zip(grads, states['exp_avg'])]
            torch._foreach_neg_(grads)
        state = self.stack_states(states)
        grad = (grads[0].unsqueeze(0) if len(params) == 1 else torch.stack(grads)).to(dtype=state['exp_avg'].dtype)
        beta1, beta2 = group['betas']
        q_left, q_right = state['q_left'], state['q_right']
        left_projected = q_left.mT @ grad
        right_projected = grad @ q_right
        projected = left_projected @ q_right

        momentum = state['exp_avg']
        if self._use_soap:
            momentum.lerp_(projected, weight=1.0 - beta1)
            state['exp_avg_sq'].lerp_(projected.square(), weight=1.0 - beta2)
            update = momentum / state['exp_avg_sq'].sqrt().add_(group['eps'])
        else:
            momentum.lerp_(grad, weight=1.0 - beta1)
            scale = state['inv_left'].unsqueeze(-1) * state['inv_right'].unsqueeze(-2)
            update = (q_left.mT @ momentum @ q_right) * (scale / scale.mul(group['eps']).add_(1.0))
        update = q_left @ update @ q_right.mT

        self.update_matrix_statistics(group, state, left_projected, right_projected, projected)
        for key in ('exp_avg', 'exp_avg_sq', 'left', 'right', 'inv_left', 'inv_right'):
            if key not in state:
                continue
            value = state[key]
            torch._foreach_copy_(states[key], list(value.unbind()))
        self._apply_updates(group, params, list(update.unbind()))

    @staticmethod
    def stack_states(states: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pack aligned state lists, using views for a single parameter."""
        return {
            key: values[0].unsqueeze(0) if len(values) == 1 else torch.stack(values) for key, values in states.items()
        }

    def _refresh_bases(
        self, group: ParamGroup, params: list[torch.Tensor], states: dict[str, list[torch.Tensor]]
    ) -> None:
        keys = ['q_left', 'q_right', 'left', 'right']
        if self._use_soap:
            keys.append('exp_avg')
        state = self.stack_states({key: states[key] for key in keys})
        q_left, q_right = state['q_left'], state['q_right']
        if self._use_soap:
            momentum = q_left @ state['exp_avg'] @ q_right.mT
        left_power, right_power = state['left'] @ q_left, state['right'] @ q_right
        if q_left.shape[-1] == q_right.shape[-1]:
            q_left, q_right = self.qr_basis(torch.cat((left_power, right_power))).chunk(2)
        else:
            q_left, q_right = self.qr_basis(left_power), self.qr_basis(right_power)
        torch._foreach_copy_(states['q_left'], list(q_left.unbind()))
        torch._foreach_copy_(states['q_right'], list(q_right.unbind()))
        if self._use_soap:
            momentum = q_left.mT @ momentum @ q_right
            torch._foreach_copy_(states['exp_avg'], list(momentum.unbind()))

    def update_matrix_statistics(
        self,
        group: ParamGroup,
        state: dict[str, torch.Tensor],
        left_projected: torch.Tensor,
        right_projected: torch.Tensor,
        projected: torch.Tensor,
    ) -> None:
        """Update batched matrix factors and spectra while reusing gradient projections."""
        beta = group['shampoo_beta'] if group['shampoo_beta'] is not None else group['betas'][1]
        inv_left, inv_right = state['inv_left'].unsqueeze(-1), state['inv_right'].unsqueeze(-2)
        left_whitened, right_whitened = right_projected * inv_right, left_projected * inv_left
        state['left'].baddbmm_(left_whitened, left_whitened.mT, beta=beta, alpha=(1.0 - beta) / projected.shape[-1])
        state['right'].baddbmm_(right_whitened.mT, right_whitened, beta=beta, alpha=(1.0 - beta) / projected.shape[-2])
        left_estimate = (projected * inv_right).square().mean(dim=-1)
        right_estimate = (projected * inv_left).square().mean(dim=-2)
        maximum = group['max_preconditioner_value']
        state['inv_left'] = self.inverse_root(state['inv_left'], left_estimate, beta, maximum)
        state['inv_right'] = self.inverse_root(state['inv_right'], right_estimate, beta, maximum)

    def _step_diagonal(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: dict[str, list[torch.Tensor]],
    ) -> None:
        grads = [grad.to(dtype=momentum.dtype) for grad, momentum in zip(grads, states['exp_avg'])]
        if self.maximize:
            torch._foreach_neg_(grads)
        beta1, beta2 = group['betas']
        torch._foreach_lerp_(states['exp_avg'], grads, weight=1.0 - beta1)
        torch._foreach_lerp_(states['exp_avg_sq'], torch._foreach_mul(grads, grads), weight=1.0 - beta2)
        denominators = torch._foreach_add(torch._foreach_sqrt(states['exp_avg_sq']), group['eps'])
        self._apply_updates(group, params, list(torch._foreach_div(states['exp_avg'], denominators)))

    def _apply_updates(self, group: ParamGroup, params: list[torch.Tensor], updates: list[torch.Tensor]) -> None:
        lr = group['lr'].to(dtype=updates[0].dtype) if isinstance(group['lr'], torch.Tensor) else group['lr']
        self.apply_weight_decay_foreach(params, [], lr, group['weight_decay'], True, False)
        foreach_add_(
            params, [update.to(dtype=param.dtype) for param, update in zip(params, updates)], alpha=-lr
        )

    def _step_single(self, group: ParamGroup, p: torch.Tensor) -> None:
        state = self.state[p]
        grad = p.grad.to(dtype=state['exp_avg'].dtype)
        self.maximize_gradient(grad, maximize=self.maximize)
        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        if not state['preconditioned'] or self._use_soap:
            projected = self.project(grad, state['Q']) if state['preconditioned'] else grad
            exp_avg.lerp_(projected, weight=1.0 - beta1)
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.lerp_(projected.square(), weight=1.0 - beta2)
            update = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])
        else:
            exp_avg.lerp_(grad, weight=1.0 - beta1)
            update = self.project(exp_avg, state['Q'])
            scale = torch.ones_like(update)
            for dim, inv_root in enumerate(state['inv_roots']):
                shape = [1] * update.ndim
                shape[dim] = -1
                scale.mul_(inv_root.reshape(shape))
            update.mul_(scale / scale.mul(group['eps']).add_(1.0))

        if state['preconditioned']:
            update = self.project(update, state['Q'], backward=True)
            self.update_preconditioner(grad, state, group)

        self._apply_updates(group, [p], [update])

    def _step_batches(self, group: ParamGroup, params: list[torch.Tensor]) -> None:
        for tensors in group_tensors_by_device_and_dtype(params, [p.grad for p in params]):
            batches: dict[tuple, list[torch.Tensor]] = defaultdict(list)
            for p in tensors['params']:
                state = self.state[p]
                step = state['step'] - 1
                shape = tuple(p.shape) if state['preconditioned'] else ()
                refresh = state['preconditioned'] and (step < 10 or step % group['precondition_frequency'] == 0)
                batches[shape, state['exp_avg'].dtype, refresh].append(p)

            for (shape, _, refresh), batch in batches.items():
                keys = ['exp_avg', 'exp_avg_sq'] if self._use_soap or not shape else ['exp_avg']
                states = {key: [self.state[p][key] for p in batch] for key in keys}
                grads = [p.grad for p in batch]
                if shape:
                    for key, source, axis in (
                        ('left', 'GG', 0),
                        ('right', 'GG', 1),
                        ('q_left', 'Q', 0),
                        ('q_right', 'Q', 1),
                        ('inv_left', 'inv_roots', 0),
                        ('inv_right', 'inv_roots', 1),
                    ):
                        states[key] = [self.state[p][source][axis] for p in batch]
                    self._get_step(self._step_foreach)(group, batch, grads, states)
                    if refresh:
                        self._get_step(self._refresh_bases)(group, batch, states)
                else:
                    self._get_step(self._step_diagonal)(group, batch, grads, states)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            use_foreach = self.can_use_foreach(group, group.get('foreach'))
            params = []
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] += 1
                if state['preconditioned'] and state['step'] == 1:
                    continue

                if use_foreach and (p.ndim == 2 or not state['preconditioned']):
                    params.append(p)
                else:
                    self._step_single(group, p)

            self._step_batches(group, params)

        return loss

class KLSOAP(KLShampoo):
    """Adam updates in the eigenbases estimated by KL-Shampoo.

    Accepts the same arguments as `KLShampoo`.

    """

    _use_soap: bool = True
