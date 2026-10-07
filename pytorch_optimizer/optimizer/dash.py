import math
from collections.abc import Iterator

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class DASH(BaseOptimizer):
    """Accelerated Shampoo with batched blocks and Newton-Denman-Beavers inverse roots.

    Implements local, layerwise DASH from https://arxiv.org/abs/2602.02016. Equal-sized blocks, including left and
    right factors, share batched storage. Scalars and squeezed vectors use one-sided inverse-square-root
    preconditioning; higher-order tensors are flattened after the first non-singleton dimension.
    Adam grafting rescales each block independently. Optimizer states use at least float32 precision.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient EMA and Shampoo statistics.
        grafting_beta: Decay rate for Adam grafting statistics. `None` uses `beta2`.
        weight_decay: Decoupled weight decay coefficient.
        block_size: Maximum block dimension. Edge blocks are processed without padding.
        precondition_frequency: Number of steps between inverse-root updates.
        start_preconditioning_step: First step using Shampoo. Earlier steps use Adam grafting.
        inverse_root_method: Inverse-root solver: `'newton_db'` or `'eigh'`.
        matrix_scaling: Newton-DB scaling: `'power'` or `'frobenius'`.
        newton_steps: Number of iterations per Newton-DB square-root computation.
        power_iteration_steps: Number of iterations for spectral-radius estimation.
        power_iteration_vectors: Number of parallel starting vectors for power iteration.
        momentum: Momentum coefficient for the grafted update.
        nesterov: Whether to use Nesterov update momentum.
        correct_bias: Whether to correct bias in the Adam grafting direction.
        eps: Term added to the Adam grafting denominator.
        matrix_eps: Diagonal regularization for inverse roots. Must be positive.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.95),
        grafting_beta: float | None = None,
        weight_decay: float = 0.0,
        block_size: int = 1024,
        precondition_frequency: int = 10,
        start_preconditioning_step: int = 1,
        inverse_root_method: str = 'newton_db',
        matrix_scaling: str = 'power',
        newton_steps: int = 10,
        power_iteration_steps: int = 10,
        power_iteration_vectors: int = 16,
        momentum: float = 0.0,
        nesterov: bool = True,
        correct_bias: bool = True,
        eps: float = 1e-8,
        matrix_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(betas[1] if grafting_beta is None else grafting_beta, 'grafting_beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')
        self.validate_positive(matrix_eps, 'matrix_eps')
        self.validate_options(inverse_root_method, 'inverse_root_method', ['newton_db', 'eigh'])
        self.validate_options(matrix_scaling, 'matrix_scaling', ['power', 'frobenius'])
        for name, value in (
            ('block_size', block_size),
            ('precondition_frequency', precondition_frequency),
            ('start_preconditioning_step', start_preconditioning_step),
            ('newton_steps', newton_steps),
            ('power_iteration_steps', power_iteration_steps),
            ('power_iteration_vectors', power_iteration_vectors),
        ):
            self.validate_positive(value, name)

        self.maximize = maximize
        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'grafting_beta': grafting_beta,
            'weight_decay': weight_decay,
            'block_size': block_size,
            'precondition_frequency': precondition_frequency,
            'start_preconditioning_step': start_preconditioning_step,
            'inverse_root_method': inverse_root_method,
            'matrix_scaling': matrix_scaling,
            'newton_steps': newton_steps,
            'power_iteration_steps': power_iteration_steps,
            'power_iteration_vectors': power_iteration_vectors,
            'momentum': momentum,
            'nesterov': nesterov,
            'correct_bias': correct_bias,
            'eps': eps,
            'matrix_eps': matrix_eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DASH'

    def _restore_state_types(self, value, saved_value):
        if isinstance(saved_value, torch.Tensor) and saved_value.is_floating_point():
            return saved_value.to(device=value.device)
        return super()._restore_state_types(value, saved_value)

    @staticmethod
    def partition(grad: torch.Tensor, block_size: int) -> Iterator[tuple[tuple[int, int, int, int], torch.Tensor]]:
        """Yield up to four batches of equal-sized blocks and their matrix bounds."""
        rows, cols = grad.shape
        row_start = 0
        for row_size in (rows // block_size * block_size, rows % block_size):
            col_start = 0
            for col_size in (cols // block_size * block_size, cols % block_size):
                if row_size and col_size:
                    height, width = min(row_size, block_size), min(col_size, block_size)
                    region = grad[row_start : row_start + row_size, col_start : col_start + col_size]
                    blocks = region.reshape(row_size // height, height, col_size // width, width)
                    yield (row_start, col_start, row_size, col_size), blocks.transpose(1, 2).reshape(-1, height, width)
                col_start += col_size
            row_start += row_size

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise NoSparseGradientError(str(self))
            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['blocks'] = []
                shape = p.squeeze().shape
                state['shape'] = (shape[0], math.prod(shape[1:])) if len(shape) > 1 else (p.numel(), 1)
                state['one_sided'] = len(shape) < 2
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                for _, block in self.partition(p.grad.reshape(state['shape']), group['block_size']):
                    batch, rows, cols = block.shape
                    sizes = (
                        [(batch, rows, rows)]
                        if state['one_sided']
                        else (
                            [(2 * batch, rows, rows)] if rows == cols else [(batch, rows, rows), (batch, cols, cols)]
                        )
                    )
                    block_state = {
                        'exp_avg_sq': torch.zeros_like(block, dtype=dtype),
                        'statistics': [torch.zeros(size, device=p.device, dtype=dtype) for size in sizes],
                        'inverse_roots': [torch.zeros(size, device=p.device, dtype=dtype) for size in sizes],
                    }
                    if group['betas'][0] > 0.0:
                        block_state['exp_avg'] = torch.zeros_like(block, dtype=dtype)
                    if group['momentum'] > 0.0:
                        block_state['momentum'] = torch.zeros_like(block, dtype=dtype)
                    state['blocks'].append(block_state)

    @staticmethod
    def matrix_scale(matrix: torch.Tensor, group: ParamGroup) -> torch.Tensor:
        """Estimate batched matrix scales using the reference's bfloat16 power iteration."""
        low_precision = matrix.to(torch.bfloat16)
        if group['matrix_scaling'] == 'frobenius':
            return torch.linalg.vector_norm(low_precision, dim=(-2, -1), keepdim=True).to(matrix.dtype)

        low_precision.diagonal(dim1=-2, dim2=-1).add_(1e-6)
        vectors = torch.randn(
            (*matrix.shape[:2], group['power_iteration_vectors']), device=matrix.device, dtype=torch.bfloat16
        )
        tiny = torch.finfo(vectors.dtype).tiny
        vectors.div_(torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min_(tiny))
        product = torch.empty_like(vectors)
        for _ in range(group['power_iteration_steps']):
            torch.bmm(low_precision, vectors, out=product)
            torch.div(product, torch.linalg.vector_norm(product, dim=1, keepdim=True).clamp_min_(tiny), out=vectors)
        torch.bmm(low_precision, vectors, out=product)
        return (vectors * product).sum(dim=1).amax(dim=1).to(matrix.dtype).view(-1, 1, 1).mul_(2.0)

    @staticmethod
    def newton_db(matrix: torch.Tensor, scale: torch.Tensor, steps: int, inverse: bool) -> torch.Tensor:
        """Compute a batched square root or inverse square root with fixed Newton-DB iterations."""
        y = matrix / scale
        correction = y.mul(-0.5)
        correction.diagonal(dim1=-2, dim2=-1).add_(1.5)
        z = correction.clone()
        if steps > 1 or not inverse:
            y = y @ correction
        scratch = torch.empty_like(y)
        for _ in range(1, steps - 1):
            torch.bmm(z, y, out=correction)
            correction.mul_(-0.5).diagonal(dim1=-2, dim2=-1).add_(1.5)
            torch.bmm(y, correction, out=scratch)
            y, scratch = scratch, y
            torch.bmm(correction, z, out=scratch)
            z, scratch = scratch, z

        if steps > 1:
            torch.bmm(z, y, out=correction)
            correction.mul_(-0.5).diagonal(dim1=-2, dim2=-1).add_(1.5)
            torch.bmm(correction, z, out=scratch) if inverse else torch.bmm(y, correction, out=scratch)
            result = scratch
        else:
            result = z if inverse else y
        return result.div_(scale.sqrt()) if inverse else result.mul_(scale.sqrt())

    def inverse_root(self, matrix: torch.Tensor, root: int, group: ParamGroup) -> torch.Tensor:
        """Compute regularized batched inverse roots without modifying the statistics."""
        regularized = matrix.clone()
        regularized.diagonal(dim1=-2, dim2=-1).add_(group['matrix_eps'])
        if group['inverse_root_method'] == 'eigh':
            values, vectors = torch.linalg.eigh(regularized)
            # Match Distributed Shampoo's spectral shift after regularized eigendecomposition.
            values.add_(group['matrix_eps'] - values.amin(dim=-1, keepdim=True).clamp_max_(0.0)).pow_(-1.0 / root)
            return (vectors * values.unsqueeze(-2)) @ vectors.transpose(-2, -1)

        scale = self.matrix_scale(regularized, group).clamp_min_(torch.finfo(matrix.dtype).tiny)
        if root == 4:
            regularized = self.newton_db(regularized, scale, group['newton_steps'], inverse=False)
            scale = scale.sqrt()
        return self.newton_db(regularized, scale, group['newton_steps'], inverse=True)

    def update_block(
        self, grad: torch.Tensor, state: dict, step: int, one_sided: bool, group: ParamGroup
    ) -> torch.Tensor:
        """Update statistics and return a grafted, optionally momentum-filtered block batch."""
        beta1, beta2 = group['betas']
        grafting_beta = beta2 if group['grafting_beta'] is None else group['grafting_beta']
        batch = grad.shape[0]
        grad = grad.to(state['exp_avg_sq'].dtype)
        statistics, inverse_roots = state['statistics'], state['inverse_roots']
        statistics[0][:batch].baddbmm_(grad, grad.transpose(1, 2), beta=beta2, alpha=1.0 - beta2)
        if not one_sided:
            statistics[-1][-batch:].baddbmm_(grad.transpose(1, 2), grad, beta=beta2, alpha=1.0 - beta2)
        state['exp_avg_sq'].mul_(grafting_beta).addcmul_(grad, grad, value=1.0 - grafting_beta)
        if beta1 > 0.0:
            grad = state['exp_avg'].lerp_(grad, weight=1.0 - beta1)

        bias1 = self.debias(beta1, step) if group['correct_bias'] else 1.0
        bias2 = self.debias(grafting_beta, step) if group['correct_bias'] else 1.0
        denom = state['exp_avg_sq'].div(bias2).sqrt_().add_(group['eps'])
        graft = grad.div(bias1).div_(denom.clamp_min_(torch.finfo(denom.dtype).tiny))
        del denom
        start = group['start_preconditioning_step']
        if step >= start:
            graft_norm = torch.linalg.vector_norm(graft, dim=(1, 2), keepdim=True)
            del graft
            if step == start or step % group['precondition_frequency'] == 0:
                for statistic, inverse_root in zip(statistics, inverse_roots):
                    inverse_root.copy_(self.inverse_root(statistic, 2 if one_sided else 4, group))
            update = inverse_roots[0][:batch] @ grad
            if not one_sided:
                update = update @ inverse_roots[-1][-batch:]
            update.mul_(graft_norm / torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True).add_(1e-16))
        else:
            update = graft

        if group['momentum'] > 0.0:
            momentum = state['momentum'].mul_(group['momentum']).add_(update)
            update = update.add_(momentum, alpha=group['momentum']) if group['nesterov'] else momentum
        return update

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1
                grad = p.grad.reshape(state['shape'])
                grad = -grad if self.maximize else grad
                update = torch.empty(state['shape'], device=p.device, dtype=p.dtype)
                for (bounds, block), block_state in zip(self.partition(grad, group['block_size']), state['blocks']):
                    direction = self.update_block(block, block_state, state['step'], state['one_sided'], group)
                    row, col, rows, cols = bounds
                    height, width = block.shape[1:]
                    direction = direction.reshape(rows // height, cols // width, height, width)
                    update[row : row + rows, col : col + cols].copy_(direction.transpose(1, 2).reshape(rows, cols))

                self.apply_weight_decay(
                    p, None, group['lr'], group['weight_decay'], weight_decouple=True, fixed_decay=False
                )
                p.add_(update.reshape_as(p), alpha=-group['lr'])
        return loss
