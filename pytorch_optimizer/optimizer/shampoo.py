import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.shampoo_utils import (
    LayerWiseGrafting,
    PreConditioner,
    PreConditionerType,
    build_graft,
    compute_power_svd,
)


class Shampoo(BaseOptimizer):
    r"""Preconditioned Stochastic Tensor Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param preconditioning_compute_steps: int. performance tuning params for controlling memory and compute
        requirements. How often to compute pre-conditioner.
    :param matrix_eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        preconditioning_compute_steps: int = 1,
        matrix_eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_step(preconditioning_compute_steps, 'preconditioning_compute_steps')
        self.validate_non_negative(matrix_eps, 'matrix_eps')

        self.preconditioning_compute_steps = preconditioning_compute_steps

        defaults: DEFAULTS = {
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

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    if momentum > 0.0:
                        state['momentum_buffer'] = grad.clone()

                    for dim_id, dim in enumerate(grad.size()):
                        state[f'pre_cond_{dim_id}'] = group['matrix_eps'] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f'inv_pre_cond_{dim_id}'] = grad.new(dim, dim).zero_()

                if momentum > 0.0:
                    grad.mul_(1.0 - momentum).add_(state['momentum_buffer'], alpha=momentum)

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
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
    r"""Scalable Preconditioned Stochastic Tensor Optimization.

        This version of Scalable Shampoo Optimizer aims for a single GPU environment, not for a distributed environment
        or XLA devices. So, the original intention is to compute pre-conditioners asynchronously on the distributed
        CPUs, but this implementation calculates them which takes 99% of the optimization time on a GPU synchronously.

        Still, it is much faster than the previous Shampoo Optimizer because using coupled Newton iteration when
        computing G^{-1/p} matrices while the previous one uses SVD which is really slow.

        Also, this implementation offers
            1. lots of plug-ins (e.g. gradient grafting, type of pre-conditioning, etc)
            2. not-yet implemented features in the official Pytorch code.
            3. readable, organized, clean code.

        Reference : https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/shampoo.py.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. beta1, beta2.
    :param moving_average_for_momentum: bool. perform moving_average for momentum (beta1).
    :param weight_decay: float. weight decay (L2 penalty).
    :param decoupled_weight_decay: bool. use decoupled weight_decay.
    :param decoupled_learning_rate: bool. use decoupled lr, otherwise couple it w/ preconditioned gradient.
    :param inverse_exponent_override: int. fixed exponent for pre-conditioner, if > 0.
    :param start_preconditioning_step: int.
    :param preconditioning_compute_steps: int. performance tuning params for controlling memory and compute
        requirements. How often to compute pre-conditioner. Ideally, 1 is the best. However, the current implementation
        doesn't work on the distributed environment (there are no statistics & pre-conditioners sync among replicas),
        compute on the GPU (not CPU) and the precision is fp32 (not fp64).
        Also, followed by the paper, `preconditioning_compute_steps` does not have a significant effect on the
        performance. So, If you have a problem with the speed, try to set this step bigger (e.g. 1000).
    :param statistics_compute_steps: int. How often to compute statistics. usually set to 1 (or 10).
    :param block_size: int. Block size for large layers (if > 0).
        Block size = 1 ==> AdaGrad (Don't do this, extremely inefficient!)
        Block size should be as large as feasible under memory/time constraints.
    :param skip_preconditioning_rank_lt: int. Skips preconditioning for parameters with rank less than this value.
    :param no_preconditioning_for_layers_with_dim_gt: int. avoid preconditioning large layers to reduce overall memory.
    :param shape_interpretation: bool. Automatic shape interpretation (for eg: [4, 3, 1024, 512] would
        result in 12 x [1024, 512] L and R statistics. Disabled by default which results in Shampoo constructing
        statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
    :param graft_type: int. type of grafting (SGD or AdaGrad or RMSProp or SQRT_N or None).
    :param pre_conditioner_type: int. type of pre-conditioner.
    :param nesterov: bool. Nesterov momentum.
    :param diagonal_eps: float. term added to the denominator to improve numerical stability.
    :param matrix_eps: float. term added to the denominator to improve numerical stability.
    :param use_svd: bool. use SVD instead of Schur-Newton method to calculate M^{-1/p}.
        Theoretically, Schur-Newton method is faster than SVD method. However, the inefficiency of the loop code and
        proper svd kernel, SVD is much faster in some cases (usually in case of small models).
        see https://github.com/kozistr/pytorch_optimizer/pull/103
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
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

        defaults: DEFAULTS = {
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

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['momentum'] = torch.zeros_like(p)
                state['pre_conditioner'] = PreConditioner(
                    p,
                    group['betas'][1],  # beta2
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
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            is_precondition_step: bool = self.is_precondition_step(group['step'])
            pre_conditioner_multiplier: float = 1.0 if group['decoupled_learning_rate'] else group['lr']

            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
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
                        g,
                        group['lr'],
                        group['weight_decay'],
                        group['decoupled_weight_decay'],
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
