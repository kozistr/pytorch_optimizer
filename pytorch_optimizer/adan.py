import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.gc import centralize_gradient
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Adan(Optimizer, BaseOptimizer):
    """
    Reference : x
    Example :
        from pytorch_optimizer import Adan
        ...
        model = YourModel()
        optimizer = Adan(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.98, 0.92, 0.99),
        weight_decay: float = 0.02,
        use_gc: bool = False,
        eps: float = 1e-16,
    ):
        """Adan optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param use_gc: bool. use gradient centralization
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.use_gc = use_gc
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)
                state['exp_avg_nest'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

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
                    raise RuntimeError('Adan does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    state['exp_avg_nest'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg, exp_avg_var, exp_avg_nest = state['exp_avg'], state['exp_avg_var'], state['exp_avg_nest']
                prev_grad = state['previous_grad']

                state['step'] += 1
                beta1, beta2, beta3 = group['betas']

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                grad_diff = grad - prev_grad
                state['previous_grad'] = grad.clone()

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_var.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)
                exp_avg_nest.mul_(beta3).add_((grad + beta2 * grad_diff) ** 2, alpha=1.0 - beta3)

                step_size = group['lr'] / math.sqrt(exp_avg_nest + self.eps)

                p.sub_(exp_avg + beta2 * exp_avg_var, alpha=step_size)
                p.div_(1.0 + group['weight_decay'])

        return loss
