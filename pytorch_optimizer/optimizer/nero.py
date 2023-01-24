import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import neuron_mean, neuron_norm


class Nero(Optimizer, BaseOptimizer):
    """Learning by Turning: Neural Architecture Aware Optimisation

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param constraints: bool.
    """

    def __init__(self, params: PARAMETERS, lr: float = 0.01, beta: float = 0.999, constraints: bool = True):
        self.lr = lr
        self.beta = beta

        self.validate_parameters()

        defaults: DEFAULTS = dict(lr=lr, constraints=constraints)
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_beta(self.beta)

    @property
    def __name__(self) -> str:
        return 'Nero'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p))

                state = self.state[p]

                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['scale'] = neuron_norm(p).mean()

                if state['scale'] == 0.0:
                    state['scale'] = 0.01

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
                    raise NoSparseGradientError(self.__name__)

                state = self.state[p]
                if len(state) == 0:
                    if group['constraints'] and p.dim() > 1:
                        p.sub_(neuron_mean(p))
                        p.div_(neuron_norm(p))

                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                    state['scale'] = neuron_norm(p).mean()
                    if state['scale'] == 0.0:
                        state['scale'] = 0.01

                state['step'] += 1

                bias_correction: float = 1.0 - self.beta ** state['step']
                state['exp_avg_sq'] = self.beta * state['exp_avg_sq'] + (1.0 - self.beta) * neuron_norm(grad) ** 2

                grad_normed = grad / (state['exp_avg_sq'] / bias_correction).sqrt()
                grad_normed[torch.isnan(grad_normed)] = 0.0

                p.sub_(group['lr'] * state['scale'] * grad_normed)

                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p))

        return loss
