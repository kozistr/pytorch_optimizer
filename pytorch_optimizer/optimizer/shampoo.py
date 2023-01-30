import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.shampoo_utils import AdagradGraft, Graft, LayerWiseGrafting, PreConditioner, SGDGraft


class Shampoo(Optimizer, BaseOptimizer):
    r"""Preconditioned Stochastic Tensor Optimization.
        Reference : https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/shampoo.py

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param beta2: float. beta2.
    :param weight_decay: float. weight decay (L2 penalty).
    :param inverse_exponent_override: int. fixed exponent for pre-conditioner, if > 0.
    :param start_preconditioning_step: int.
    :param preconditioning_compute_steps: int. performance tuning params for controlling memory and compute
        requirements. How often to compute pre-conditioner.
    :param statistics_compute_steps: int. How often to compute statistics.
    :param block_size: int. Block size for large layers (if > 0).
        Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
        Block size should be as large as feasible under memory/time constraints.
    :param shape_interpretation: bool. Automatic shape interpretation (for eg: [4, 3, 1024, 512] would
        result in 12 x [1024, 512] L and R statistics. Disabled by default which results in Shampoo constructing
        statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
    :param graft_type: bool. Type of grafting (SGD or AdaGrad).
    :param nesterov: bool. Nesterov momentum.
    :param diagonal_eps: float. term added to the denominator to improve numerical stability.
    :param matrix_eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.0,
        beta2: float = 1.0,
        weight_decay: float = 0.0,
        inverse_exponent_override: int = 0,
        start_preconditioning_step: int = 1,
        preconditioning_compute_steps: int = 1,
        statistics_compute_steps: int = 1,
        block_size: int = 128,
        shape_interpretation: bool = True,
        graft_type: int = LayerWiseGrafting.SGD,
        nesterov: bool = True,
        diagonal_eps: float = 1e-6,
        matrix_eps: float = 1e-12,
    ):
        self.lr = lr
        self.momentum = momentum
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.inverse_exponent_override = inverse_exponent_override
        self.start_preconditioning_step = start_preconditioning_step
        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.statistics_compute_steps = statistics_compute_steps
        self.block_size = block_size
        self.shape_interpretation = shape_interpretation
        self.graft_type = graft_type
        self.nesterov = nesterov
        self.diagonal_eps = diagonal_eps
        self.matrix_eps = matrix_eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_weight_decay(self.weight_decay)
        self.validate_update_frequency(self.start_preconditioning_step)
        self.validate_update_frequency(self.statistics_compute_steps)
        self.validate_update_frequency(self.preconditioning_compute_steps)
        self.validate_epsilon(self.diagonal_eps)
        self.validate_epsilon(self.matrix_eps)

    @property
    def __str__(self) -> str:
        return 'Shampoo'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['momentum'] = torch.zeros_like(p)
                state['pre_conditioner'] = PreConditioner(
                    p,
                    self.beta2,
                    self.inverse_exponent_override,
                    self.block_size,
                    self.shape_interpretation,
                    self.matrix_eps,
                )
                if self.graft_type == LayerWiseGrafting.ADAGRAD:
                    state['graft'] = AdagradGraft(p, self.diagonal_eps)
                elif self.graft_type == LayerWiseGrafting.SGD:
                    state['graft'] = SGDGraft(p)
                else:
                    state['graft'] = Graft(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__str__)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['pre_conditioner'] = PreConditioner(
                        p,
                        self.beta2,
                        self.inverse_exponent_override,
                        self.block_size,
                        self.shape_interpretation,
                        self.matrix_eps,
                    )
                    if self.graft_type == LayerWiseGrafting.ADAGRAD:
                        state['graft'] = AdagradGraft(p, self.diagonal_eps)
                    elif self.graft_type == LayerWiseGrafting.SGD:
                        state['graft'] = SGDGraft(p)
                    else:
                        state['graft'] = Graft(p)

                state['step'] += 1
                pre_conditioner, graft = state['pre_conditioner'], state['graft']

                # gather statistics, compute pre-conditioners
                graft.add_statistics(grad)
                if state['step'] % self.statistics_compute_steps == 0:
                    pre_conditioner.add_statistics(grad)
                if state['step'] % self.preconditioning_compute_steps == 0:
                    pre_conditioner.compute_pre_conditioners()

                # pre-condition gradients
                graft_grad: torch.Tensor = graft.precondition_gradient(grad)
                shampoo_grad: torch.Tensor = grad
                if state['step'] >= self.start_preconditioning_step:
                    shampoo_grad = pre_conditioner.preconditioned_grad(grad)

                # grafting
                graft_norm = torch.norm(graft_grad)
                shampoo_norm = torch.norm(shampoo_grad)
                shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

                # apply weight decay (adam style)
                if group['weight_decay'] > 0.0:
                    shampoo_grad.add_(p, alpha=group['weight_decay'])
                    graft_grad.add_(p, alpha=group['weight_decay'])

                # Momentum and Nesterov momentum, if needed
                state['momentum'].mul_(group['momentum']).add_(shampoo_grad)
                graft_momentum = graft.update_momentum(grad, group['momentum'])

                if state['step'] >= self.start_preconditioning_step:
                    momentum_update = state['momentum']
                    wd_update = shampoo_grad
                else:
                    momentum_update = graft_momentum
                    wd_update = graft_grad

                if self.nesterov:
                    momentum_update.mul_(group['momentum']).add_(wd_update)

                p.add_(momentum_update, alpha=-group['lr'])

        return loss
