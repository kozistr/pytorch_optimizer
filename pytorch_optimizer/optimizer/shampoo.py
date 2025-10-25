import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.shampoo_utils import (
    LayerWiseGrafting,
    PreConditioner,
    PreConditionerType,
    build_graft,
    compute_power_svd,
)


class Shampoo(BaseOptimizer):
    """Preconditioned Stochastic Tensor Optimization.

    Args:
        params (Parameters): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        momentum (float): momentum factor.
        weight_decay (float): weight decay (L2 penalty).
        weight_decouple (bool): optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): fix weight decay.
        preconditioning_compute_steps (int): how often to compute the preconditioner,
            tuning memory and compute requirements.
        matrix_eps (float): term added to denominator to improve numerical stability.
        maximize (bool): maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        preconditioning_compute_steps: int = 1,
        matrix_eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_step(preconditioning_compute_steps, 'preconditioning_compute_steps')
        self.validate_non_negative(matrix_eps, 'matrix_eps')

        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'matrix_eps': matrix_eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Shampoo'

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
                if group['momentum'] > 0.0:
                    state['momentum_buffer'] = grad.clone()

                for dim_id, dim in enumerate(grad.size()):
                    state[f'pre_cond_{dim_id}'] = group['matrix_eps'] * torch.eye(dim, out=grad.new(dim, dim))
                    state[f'inv_pre_cond_{dim_id}'] = grad.new(dim, dim).zero_()

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

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if momentum > 0.0:
                    grad.mul_(1.0 - momentum).add_(state['momentum_buffer'], alpha=momentum)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                order: int = grad.ndimension()
                original_size: int = grad.size()
                for dim_id, dim in enumerate(grad.size()):
                    pre_cond, inv_pre_cond = state[f'pre_cond_{dim_id}'], state[f'inv_pre_cond_{dim_id}']

                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()

                    grad = grad.view(dim, -1)
                    grad_t = grad.t()

                    pre_cond.add_(grad @ grad_t)
                    if group['step'] % self.preconditioning_compute_steps == 0:
                        inv_pre_cond.copy_(compute_power_svd(pre_cond, order))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_pre_cond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_pre_cond @ grad
                        grad = grad.view(transposed_size)

                state['momentum_buffer'] = grad

                p.add_(grad, alpha=-group['lr'])

        return loss


class ScalableShampoo(BaseOptimizer):
    """Scalable Preconditioned Stochastic Tensor Optimization.

    This version of the Scalable Shampoo Optimizer targets single GPU environments,
    computing pre-conditioners synchronously on GPU (which takes most of the optimization time).
    It is faster than previous Shampoo implementations by using coupled Newton iteration
    for matrix inverse powers instead of slow SVD calculations.

    Features include:
    1. Various plug-ins (e.g., gradient grafting, preconditioning types),
    2. Additional features beyond official PyTorch code,
    3. Readable and well-organized implementation.

    Reference:
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/shampoo.py

    Args:
        params (Parameters): iterable or dicts defining parameter groups.
        lr (float): learning rate.
        betas (tuple): beta1 and beta2 for momentum.
        moving_average_for_momentum (bool): whether to perform moving average for momentum (beta1).
        weight_decay (float): weight decay (L2 penalty).
        decoupled_weight_decay (bool): use decoupled weight decay.
        decoupled_learning_rate (bool): use decoupled learning rate, otherwise coupled with preconditioned gradient.
        inverse_exponent_override (int): fixed exponent for preconditioner if > 0.
        start_preconditioning_step (int): step to start preconditioning.
        preconditioning_compute_steps (int): frequency of preconditioner computation.
        statistics_compute_steps (int): frequency of statistics computation.
        block_size (int): block size for large layers; 1 means AdaGrad (inefficient).
        skip_preconditioning_rank_lt (int): skip preconditioning for layers with rank below this.
        no_preconditioning_for_layers_with_dim_gt (int): avoid preconditioning large layers.
        shape_interpretation (bool): automatic shape interpretation for tensor dims.
        graft_type (int): type of grafting (SGD, AdaGrad, RMSProp, etc.).
        pre_conditioner_type (int): type of preconditioner.
        nesterov (bool): enable Nesterov momentum.
        diagonal_eps (float): epsilon for numerical stability in diagonal.
        matrix_eps (float): epsilon for numerical stability in matrix.
        use_svd (bool): whether to use SVD for matrix inverse powers (alternative is Schur-Newton).
        maximize (bool): maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        moving_average_for_momentum: bool = False,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        decoupled_learning_rate: bool = True,
        inverse_exponent_override: int = 0,
        start_preconditioning_step: int = 25,
        preconditioning_compute_steps: int = 1000,
        statistics_compute_steps: int = 1,
        block_size: int = 512,
        skip_preconditioning_rank_lt: int = 1,
        no_preconditioning_for_layers_with_dim_gt: int = 8192,
        shape_interpretation: bool = True,
        graft_type: int = LayerWiseGrafting.SGD,
        pre_conditioner_type: int = PreConditionerType.ALL,
        nesterov: bool = True,
        diagonal_eps: float = 1e-10,
        matrix_eps: float = 1e-6,
        use_svd: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_step(start_preconditioning_step, 'start_preconditioning_step')
        self.validate_step(preconditioning_compute_steps, 'preconditioning_compute_steps')
        self.validate_step(statistics_compute_steps, 'statistics_compute_steps')
        self.validate_non_negative(diagonal_eps, 'diagonal_eps')
        self.validate_non_negative(matrix_eps, 'matrix_eps')

        self.inverse_exponent_override = inverse_exponent_override
        self.start_preconditioning_step = start_preconditioning_step
        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.statistics_compute_steps = statistics_compute_steps
        self.block_size = block_size
        self.skip_preconditioning_rank_lt = skip_preconditioning_rank_lt
        self.no_preconditioning_for_layers_with_dim_gt = no_preconditioning_for_layers_with_dim_gt
        self.shape_interpretation = shape_interpretation
        self.graft_type = graft_type
        self.pre_conditioner_type = pre_conditioner_type
        self.diagonal_eps = diagonal_eps
        self.matrix_eps = matrix_eps
        self.use_svd = use_svd
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'decoupled_weight_decay': decoupled_weight_decay,
            'decoupled_learning_rate': decoupled_learning_rate,
            'moving_average_for_momentum': moving_average_for_momentum,
            'nesterov': nesterov,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ScalableShampoo'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        _, beta2 = group['betas']

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
                state['momentum'] = torch.zeros_like(grad)
                state['pre_conditioner'] = PreConditioner(
                    p,
                    beta2,
                    self.inverse_exponent_override,
                    self.block_size,
                    self.skip_preconditioning_rank_lt,
                    self.no_preconditioning_for_layers_with_dim_gt,
                    self.shape_interpretation,
                    self.pre_conditioner_type,
                    self.matrix_eps,
                    self.use_svd,
                )
                state['graft'] = build_graft(p, self.graft_type, self.diagonal_eps)

    def is_precondition_step(self, step: int) -> bool:
        return step >= self.start_preconditioning_step

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

            is_precondition_step: bool = self.is_precondition_step(group['step'])
            pre_conditioner_multiplier: float = 1.0 if group['decoupled_learning_rate'] else group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                pre_conditioner, graft = state['pre_conditioner'], state['graft']

                graft.add_statistics(grad, beta2)
                if group['step'] % self.statistics_compute_steps == 0:
                    pre_conditioner.add_statistics(grad)
                if group['step'] % self.preconditioning_compute_steps == 0:
                    pre_conditioner.compute_pre_conditioners()

                graft_grad: torch.Tensor = graft.precondition_gradient(grad * pre_conditioner_multiplier)
                shampoo_grad: torch.Tensor = (
                    pre_conditioner.preconditioned_grad(grad) if is_precondition_step else grad
                )

                if self.graft_type != LayerWiseGrafting.NONE:
                    graft_norm = torch.linalg.norm(graft_grad)
                    shampoo_norm = torch.linalg.norm(shampoo_grad)

                    shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

                for g in (graft_grad, shampoo_grad):
                    self.apply_weight_decay(
                        p,
                        grad=g,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['decoupled_weight_decay'],
                        fixed_decay=False,
                    )

                state['momentum'].mul_(beta1).add_(shampoo_grad)
                graft_momentum = graft.update_momentum(grad, beta1)

                momentum_update = state['momentum'] if is_precondition_step else graft_momentum

                if group['nesterov']:
                    w: float = (1.0 - beta1) if group['moving_average_for_momentum'] else 1.0

                    wd_update = shampoo_grad if is_precondition_step else graft_grad
                    wd_update.mul_(w)

                    momentum_update.mul_(beta1).add_(wd_update)

                p.add_(momentum_update, alpha=-group['lr'])

        return loss
