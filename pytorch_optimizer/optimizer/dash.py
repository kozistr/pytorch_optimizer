from typing import Dict, List, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT

BlockSpec = Tuple[int, int, int, int, int, int]

INV_ROOT_METHODS: List[str] = ['evd', 'newton_db']
EVD_HEURISTICS: List[str] = ['shampoo', 'abs', 'abs_add', 'relu']
MATRIX_SCALINGS: List[str] = ['fro', 'power_iter', 'power_iter_multi']


def get_block_layout(rows: int, cols: int, block_size: int) -> List[BlockSpec]:
    r"""Split a `rows x cols` matrix into groups of equally shaped blocks.

    Every group is a rectangular region made of `n_row_blocks x n_col_blocks` blocks of `block_rows x block_cols`.
    A matrix has at most four groups: the full `block_size` blocks, the remainder strip of rows, the remainder strip of
    columns, and the corner.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        block_size (int): Maximum block edge length.

    Returns:
        List of `(row_start, n_row_blocks, block_rows, col_start, n_col_blocks, block_cols)`.

    """

    def split(n: int) -> List[Tuple[int, int, int]]:
        full, rest = divmod(n, block_size)
        parts: List[Tuple[int, int, int]] = []
        if full > 0:
            parts.append((0, full, block_size))
        if rest > 0:
            parts.append((full * block_size, 1, rest))
        return parts

    return [(r0, nr, rh, c0, nc, cw) for r0, nr, rh in split(rows) for c0, nc, cw in split(cols)]


def stack_blocks(x: torch.Tensor, spec: BlockSpec) -> torch.Tensor:
    r"""Stack the blocks of one group into a tensor of shape `(n_blocks, block_rows, block_cols)`."""
    r0, nr, rh, c0, nc, cw = spec
    return x[r0 : r0 + nr * rh, c0 : c0 + nc * cw].reshape(nr, rh, nc, cw).transpose(1, 2).reshape(nr * nc, rh, cw)


def unstack_blocks(blocks: torch.Tensor, spec: BlockSpec, out: torch.Tensor) -> None:
    r"""Write the stacked blocks of one group back into `out`. Inverse of `stack_blocks`."""
    r0, nr, rh, c0, nc, cw = spec
    out[r0 : r0 + nr * rh, c0 : c0 + nc * cw] = (
        blocks.reshape(nr, nc, rh, cw).transpose(1, 2).reshape(nr * rh, nc * cw)
    )


def max_eigenvalue_power_iteration(
    matrix: torch.Tensor, num_iters: int, num_vecs: int = 1, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    r"""Estimate the largest eigenvalue of a batch of symmetric positive semi-definite matrices.

    Args:
        matrix (torch.Tensor): Batch of matrices of shape `(N, B, B)`.
        num_iters (int): Number of power iterations.
        num_vecs (int): Number of vectors iterated in parallel. The largest Rayleigh quotient is returned, which makes
            it less likely to stop on an eigenvector that does not belong to the largest eigenvalue.
        generator (Optional[torch.Generator]): Random generator for the start vectors.

    Returns:
        Tensor of shape `(N, 1, 1)`.

    """
    n, b, _ = matrix.shape

    v = torch.randn(n, b, num_vecs, device=matrix.device, dtype=matrix.dtype, generator=generator)
    v = v / v.norm(dim=1, keepdim=True)

    for _ in range(num_iters):
        av = torch.bmm(matrix, v)
        v = av / av.norm(dim=1, keepdim=True)

    rayleigh = (v * torch.bmm(matrix, v)).sum(dim=1)
    return rayleigh.max(dim=1).values.float().view(n, 1, 1)


def get_matrix_scaling(
    matrix: torch.Tensor,
    scaling: str,
    eps_power_iter: float,
    power_iter_steps: int,
    scaling_const: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Get a per-matrix scalar `s` such that the eigenvalues of `matrix / s` lie in `(0, 1)`.

    Args:
        matrix (torch.Tensor): Batch of symmetric positive definite matrices of shape `(N, B, B)`.
        scaling (str): `fro` uses the Frobenius norm. `power_iter` and `power_iter_multi` use a power iteration in
            bfloat16 and multiply the estimate by `scaling_const`.
        eps_power_iter (float): Value added to the diagonal in the power iteration. It keeps the iteration finite if a
            block is (almost) zero.
        power_iter_steps (int): Number of power iterations.
        scaling_const (float): Safety factor for the estimate of the largest eigenvalue.
        generator (Optional[torch.Generator]): Random generator for the power iteration.

    """
    matrix_16 = matrix.bfloat16()

    if scaling == 'fro':
        return matrix_16.norm(dim=(1, 2), keepdim=True).float()

    idx = torch.arange(matrix.shape[-1], device=matrix.device)
    matrix_16[:, idx, idx] += eps_power_iter

    num_vecs: int = 16 if scaling == 'power_iter_multi' else 1
    return max_eigenvalue_power_iteration(matrix_16, power_iter_steps, num_vecs, generator) * scaling_const


def newton_db_root(matrix: torch.Tensor, scale: torch.Tensor, num_steps: int, inverse: bool) -> torch.Tensor:
    r"""Coupled Newton iteration for the square root or the inverse square root of a batch of matrices.

    `Y_k -> A^(1/2)` and `Z_k -> A^(-1/2)` for `A / scale` with eigenvalues in `(0, 1)`.

    Args:
        matrix (torch.Tensor): Batch of symmetric positive definite matrices of shape `(N, B, B)`.
        scale (torch.Tensor): Scaling of shape `(N, 1, 1)`.
        num_steps (int): Number of iterations.
        inverse (bool): Return the inverse square root instead of the square root.

    """
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)

    a = matrix / scale
    e = 1.5 * eye - 0.5 * a
    y, z = a @ e, e

    for _ in range(1, num_steps):
        e = 1.5 * eye - 0.5 * (z @ y)
        y, z = y @ e, e @ z

    sqrt_scale = scale.sqrt()
    return z / sqrt_scale if inverse else y * sqrt_scale


def matrix_inverse_fourth_root(
    matrix: torch.Tensor,
    method: str,
    eps: float,
    evd_heuristic: str = 'shampoo',
    scaling: str = 'power_iter',
    eps_power_iter: float = 1e-6,
    power_iter_steps: int = 10,
    scaling_const: float = 2.0,
    newton_steps: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Compute `(matrix + eps I)^(-1/4)` for a batch of symmetric positive semi-definite matrices.

    Args:
        matrix (torch.Tensor): Batch of matrices of shape `(N, B, B)` in float32.
        method (str): `evd` or `newton_db`.
        eps (float): Regularization.
        evd_heuristic (str): How the spectrum is post-processed for `evd`. `shampoo` is the Distributed Shampoo rule.
            `abs` and `abs_add` take the absolute value of the spectrum minus `eps`, `abs_add` adds `eps` back.
            `relu` drops eigenvalues that are not larger than `eps`, so the result has a low rank.
        scaling (str): Scaling of the input for `newton_db`.
        eps_power_iter (float): Diagonal damping of the power iteration.
        power_iter_steps (int): Number of power iterations.
        scaling_const (float): Safety factor for the estimate of the largest eigenvalue.
        newton_steps (int): Number of Newton iterations per square root.
        generator (Optional[torch.Generator]): Random generator for the power iteration.

    """
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    regularized = matrix + eps * eye

    if method == 'newton_db':
        scale = get_matrix_scaling(regularized, scaling, eps_power_iter, power_iter_steps, scaling_const, generator)
        sqrt = newton_db_root(regularized, scale, newton_steps, inverse=False)
        return newton_db_root(sqrt, scale.sqrt(), newton_steps, inverse=True)

    eigvals, eigvecs = torch.linalg.eigh(regularized)

    if evd_heuristic == 'abs':
        eigvals = (eigvals - eps).abs()
    elif evd_heuristic == 'abs_add':
        eigvals = (eigvals - eps).abs() + eps
    elif evd_heuristic == 'relu':
        eigvals = eigvals - eps
        eigvals = torch.where(eigvals < eps, torch.zeros_like(eigvals), eigvals)
    else:
        eigvals = eigvals + (eps - eigvals.min(dim=1, keepdim=True).values.clamp(max=0.0))

    if evd_heuristic == 'relu':
        inv_root = torch.where(eigvals > 0.0, eigvals.clamp(min=torch.finfo(eigvals.dtype).tiny).pow(-0.25), 0.0)
    else:
        inv_root = eigvals.pow(-0.25)

    return (eigvecs * inv_root.unsqueeze(1)) @ eigvecs.transpose(1, 2)


class DASH(BaseOptimizer):
    r"""Distributed Accelerated SHampoo, on a single device.

    Every matrix parameter is split into blocks of at most `block_size x block_size`. The left and right Shampoo
    factors of the blocks are stacked, so the statistics, the inverse roots and the preconditioning are batched matrix
    products. The inverse roots of all blocks of the same size are computed in one call, across parameters.
    The Shampoo direction is rescaled block by block to the norm of the Adam direction (Adam grafting).

    Parameters with one non-singleton dimension (biases, norms) are updated with AdamW. Parameters with more than two
    dimensions are viewed as matrices of shape `(shape[0], -1)`.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients for the gradient moving average, and for the second moment of the grafting
            direction and of AdamW.
        shampoo_beta (float): Decay of the moving average of the left and right factors. If it is 1.0, the factors are
            accumulated as in AdaGrad.
        momentum (float): Momentum on the preconditioned update.
        use_nesterov (bool): Use Nesterov momentum. Only used if `momentum` is positive.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Whether to fix weight decay.
        block_size (int): Maximum edge length of a block.
        start_preconditioning_step (int): Steps before this one use the Adam direction only.
        preconditioning_frequency (int): Number of steps between updates of the inverse roots.
        inv_root_method (str): `evd` (eigenvalue decomposition) or `newton_db` (coupled Newton iteration).
        evd_heuristic (str): Spectrum post-processing for `evd`, one of `shampoo`, `abs`, `abs_add`, `relu`.
        matrix_scaling (str): Input scaling for `newton_db`, one of `fro`, `power_iter`, `power_iter_multi`.
        matrix_scaling_steps (int): Number of power iterations for the scaling.
        matrix_scaling_const (float): Factor applied to the estimate of the largest eigenvalue.
        newton_steps (int): Number of Newton iterations for each square root.
        use_bias_correction (bool): Bias-correct the gradient moving average and the grafting second moment.
        eps (float): Regularization of the factors before the inverse root.
        eps_graft (float): Term added to the denominator of the grafting direction, and of AdamW.
        eps_power_iter (float): Diagonal damping of the power iteration.
        maximize (bool): Maximize the objective with respect to the parameters instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        momentum: float = 0.0,
        use_nesterov: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        block_size: int = 1024,
        start_preconditioning_step: int = -1,
        preconditioning_frequency: int = 10,
        inv_root_method: str = 'evd',
        evd_heuristic: str = 'shampoo',
        matrix_scaling: str = 'power_iter',
        matrix_scaling_steps: int = 10,
        matrix_scaling_const: float = 2.0,
        newton_steps: int = 10,
        use_bias_correction: bool = True,
        eps: float = 1e-10,
        eps_graft: float = 1e-8,
        eps_power_iter: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shampoo_beta, 'shampoo_beta', 0.0, 1.0, range_type='[]')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(block_size, 'block_size')
        self.validate_positive(preconditioning_frequency, 'preconditioning_frequency')
        self.validate_options(inv_root_method, 'inv_root_method', INV_ROOT_METHODS)
        self.validate_options(evd_heuristic, 'evd_heuristic', EVD_HEURISTICS)
        self.validate_options(matrix_scaling, 'matrix_scaling', MATRIX_SCALINGS)
        self.validate_positive(matrix_scaling_steps, 'matrix_scaling_steps')
        self.validate_positive(matrix_scaling_const, 'matrix_scaling_const')
        self.validate_positive(newton_steps, 'newton_steps')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(eps_graft, 'eps_graft')
        self.validate_non_negative(eps_power_iter, 'eps_power_iter')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'momentum': momentum,
            'use_nesterov': use_nesterov,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'block_size': block_size,
            'start_preconditioning_step': start_preconditioning_step,
            'preconditioning_frequency': preconditioning_frequency,
            'inv_root_method': inv_root_method,
            'evd_heuristic': evd_heuristic,
            'matrix_scaling': matrix_scaling,
            'matrix_scaling_steps': matrix_scaling_steps,
            'matrix_scaling_const': matrix_scaling_const,
            'newton_steps': newton_steps,
            'use_bias_correction': use_bias_correction,
            'eps': eps,
            'eps_graft': eps_graft,
            'eps_power_iter': eps_power_iter,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DASH'

    @staticmethod
    def get_matrix_shape(p: torch.Tensor) -> Optional[Tuple[int, int]]:
        r"""Return the 2D shape a parameter is preconditioned as, or `None` if it is updated with AdamW."""
        shape = [d for d in p.shape if d != 1]
        if len(shape) < 2:
            return None
        return shape[0], p.numel() // shape[0]

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

            if len(state) > 0:
                continue

            matrix_shape = self.get_matrix_shape(p)
            if matrix_shape is None:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                continue

            rows, cols = matrix_shape
            options = {'dtype': torch.promote_types(p.dtype, torch.float32), 'device': p.device}

            state['exp_avg_sq'] = torch.zeros(rows, cols, **options)
            if group['betas'][0] > 0.0:
                state['exp_avg'] = torch.zeros(rows, cols, **options)
            if group['momentum'] > 0.0:
                state['momentum_buffer'] = torch.zeros(rows, cols, **options)

            for i, (_, nr, rh, _, nc, cw) in enumerate(get_block_layout(rows, cols, group['block_size'])):
                state[f'left_{i}'] = torch.zeros(nr * nc, rh, rh, **options)
                state[f'right_{i}'] = torch.zeros(nr * nc, cw, cw, **options)
                state[f'inv_left_{i}'] = torch.zeros(nr * nc, rh, rh, **options)
                state[f'inv_right_{i}'] = torch.zeros(nr * nc, cw, cw, **options)

    @staticmethod
    def update_adamw(p: torch.Tensor, grad: torch.Tensor, state: Dict, group: ParamGroup) -> None:
        beta1, beta2 = group['betas']
        step: int = group['step']

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        exp_avg.lerp_(grad, weight=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1: float = BaseOptimizer.debias(beta1, step)
        bias_correction2: float = BaseOptimizer.debias(beta2, step)

        de_nom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(group['eps_graft'])
        p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)

    def update_statistics(
        self, state: Dict, grad_2d: torch.Tensor, layout: List[BlockSpec], group: ParamGroup
    ) -> None:
        r"""Update the factors, the grafting second moment, and the gradient moving average."""
        beta1, beta_graft = group['betas']
        shampoo_beta: float = group['shampoo_beta']

        for i, spec in enumerate(layout):
            blocks = stack_blocks(grad_2d, spec)
            blocks_t = blocks.transpose(1, 2)

            for name, gram in ((f'left_{i}', blocks @ blocks_t), (f'right_{i}', blocks_t @ blocks)):
                if shampoo_beta < 1.0:
                    state[name].lerp_(gram, weight=1.0 - shampoo_beta)
                else:
                    state[name].add_(gram)

        state['exp_avg_sq'].mul_(beta_graft).addcmul_(grad_2d, grad_2d, value=1.0 - beta_graft)

        if beta1 > 0.0:
            state['exp_avg'].lerp_(grad_2d, weight=1.0 - beta1)

    def update_inverse_roots(self, group: ParamGroup, matrix_states: List[Tuple[Dict, List[BlockSpec]]]) -> None:
        r"""Compute the inverse fourth roots of all factors of a group, batching the factors of the same size."""
        by_size: Dict[Tuple[torch.device, int], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        for state, layout in matrix_states:
            for i in range(len(layout)):
                for side in ('left', 'right'):
                    factor, inv_factor = state[f'{side}_{i}'], state[f'inv_{side}_{i}']
                    by_size.setdefault((factor.device, factor.shape[-1]), []).append((factor, inv_factor))

        for (device, _), pairs in by_size.items():
            inv_roots = matrix_inverse_fourth_root(
                torch.cat([factor for factor, _ in pairs]),
                method=group['inv_root_method'],
                eps=group['eps'],
                evd_heuristic=group['evd_heuristic'],
                scaling=group['matrix_scaling'],
                eps_power_iter=group['eps_power_iter'],
                power_iter_steps=group['matrix_scaling_steps'],
                scaling_const=group['matrix_scaling_const'],
                newton_steps=group['newton_steps'],
                generator=torch.Generator(device=device).manual_seed(group['step']),
            )
            for (_, inv_factor), inv_root in zip(pairs, inv_roots.split([factor.shape[0] for factor, _ in pairs])):
                inv_factor.copy_(inv_root)

    @staticmethod
    def get_direction(state: Dict, grad_2d: torch.Tensor, layout: List[BlockSpec], group: ParamGroup) -> torch.Tensor:
        r"""Shampoo direction `L^(-1/4) G R^(-1/4)` rescaled per block to the norm of the Adam direction."""
        beta1, beta_graft = group['betas']
        step: int = group['step']

        use_momentum: bool = beta1 > 0.0
        chosen = state['exp_avg'] if use_momentum else grad_2d

        bias_correction1: float = (
            BaseOptimizer.debias(beta1, step) if group['use_bias_correction'] and use_momentum else 1.0
        )
        bias_correction2: float = BaseOptimizer.debias(beta_graft, step) if group['use_bias_correction'] else 1.0

        graft = (chosen / bias_correction1) / (group['eps_graft'] + (state['exp_avg_sq'] / bias_correction2).sqrt())

        if step < max(group['start_preconditioning_step'], 1):
            return graft

        direction = torch.empty_like(graft)
        for i, spec in enumerate(layout):
            update = state[f'inv_left_{i}'] @ stack_blocks(chosen, spec) @ state[f'inv_right_{i}']

            graft_norm = stack_blocks(graft, spec).norm(dim=(1, 2), keepdim=True)
            update_norm = update.norm(dim=(1, 2), keepdim=True)

            unstack_blocks(update * (graft_norm / (update_norm + 1e-16)), spec, direction)

        return direction

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            start_step: int = max(group['start_preconditioning_step'], 1)
            precondition: bool = group['step'] >= start_step
            update_roots: bool = precondition and (
                group['step'] == start_step or group['step'] % group['preconditioning_frequency'] == 0
            )

            matrix_params: List[Tuple[torch.Tensor, torch.Tensor, Dict, List[BlockSpec]]] = []

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                self.maximize_gradient(grad, maximize=self.maximize)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                state = self.state[p]

                matrix_shape = self.get_matrix_shape(p)
                if matrix_shape is None:
                    self.update_adamw(p, grad, state, group)
                    continue

                grad_2d = grad.to(torch.promote_types(grad.dtype, torch.float32)).reshape(matrix_shape)
                layout = get_block_layout(*matrix_shape, group['block_size'])

                self.update_statistics(state, grad_2d, layout, group)
                matrix_params.append((p, grad_2d, state, layout))

            if update_roots:
                self.update_inverse_roots(group, [(state, layout) for _, _, state, layout in matrix_params])

            momentum: float = group['momentum']

            for p, grad_2d, state, layout in matrix_params:
                update = self.get_direction(state, grad_2d, layout, group)

                if momentum > 0.0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(update)
                    update = update.add_(buf, alpha=momentum) if group['use_nesterov'] else buf

                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-group['lr'])

        return loss
