import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.gc import centralize_gradient
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.utils import projection


class AdamP(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/clovaai/AdamP
    Example :
        from pytorch_optimizer import AdamP
        ...
        model = YourModel()
        optimizer = AdamP(model.parameters())
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
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        use_gc: bool = False,
        nesterov: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        """AdamP optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param delta: float. threshold that determines whether a set of parameters is scale invariant or not
        :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
            on scale-variant parameters
        :param use_gc: bool. use gradient centralization
        :param nesterov: bool. enables Nesterov momentum
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.wd_ratio = wd_ratio
        self.use_gc = use_gc

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
            adamd_debias_term=adamd_debias_term,
            eps=eps,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_ratio(self.wd_ratio)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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
                    raise RuntimeError('AdamP does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                beta1, beta2 = group['betas']

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if group['nesterov']:
                    perturb = (beta1 * exp_avg + (1.0 - beta1) * grad) / de_nom
                else:
                    perturb = exp_avg / de_nom

                wd_ratio: float = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                if group['weight_decay'] > 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_ratio)

                step_size = group['lr']
                if not group['adamd_debias_term']:
                    step_size /= bias_correction1

                p.add_(perturb, alpha=-step_size)

        return loss
