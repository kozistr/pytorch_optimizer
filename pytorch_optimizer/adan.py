import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.gc import centralize_gradient
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Adan(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/sail-sg/Adan/blob/main/adan.py
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
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        use_gc: bool = False,
        eps: float = 1e-8,
    ):
        """Adan optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param weight_decouple: bool. decoupled weight decay
        :param use_gc: bool. use gradient centralization
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.use_gc = use_gc
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
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
                state['exp_avg_diff'] = torch.zeros_like(p)
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
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['exp_avg_nest'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg, exp_avg_diff, exp_avg_nest = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_nest']
                prev_grad = state['previous_grad']

                state['step'] += 1
                beta1, beta2, beta3 = group['betas']

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                bias_correction3 = 1.0 - beta3 ** state['step']

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                grad_diff = grad - prev_grad
                state['previous_grad'] = grad.clone()

                update = grad + beta2 * grad_diff

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)
                exp_avg_nest.mul_(beta3).addcmul_(update, update, value=1.0 - beta3)

                de_nom = (exp_avg_nest.sqrt_() / math.sqrt(bias_correction3)).add_(self.eps)
                perturb = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2).div_(de_nom)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    p.add_(perturb, alpha=-group['lr'])
                else:
                    p.add_(perturb, alpha=-group['lr'])
                    p.div_(1.0 + group['lr'] * group['weight_decay'])

        return loss
