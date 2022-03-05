import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class DiffRGrad(Optimizer, BaseOptimizer):
    """
    Reference 1 : https://github.com/shivram1987/diffGrad
    Reference 2 : https://github.com/LiyuanLucasLiu/RAdam
    Reference 3 : https://github.com/lessw2020/Best-Deep-Learning-Optimizers/blob/master/diffgrad/diff_rgrad.py
    Example :
        from pytorch_optimizer import DiffRGrad
        ...
        model = YourModel()
        optimizer = DiffRGrad(model.parameters())
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
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        """Blend RAdam with DiffGrad
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param n_sma_threshold: int. (recommended is 5)
        :param degenerated_to_sgd: bool. degenerated to SGD
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adamd_debias_term=adamd_debias_term,
            buffer=[[None, None, None] for _ in range(10)],
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
                state['exp_avg_sq'] = torch.zeros_like(p)
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
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                if grad.dtype in (torch.float16, torch.bfloat16):
                    grad = grad.float()

                p_fp32 = p
                if p.dtype in (torch.float16, torch.bfloat16):
                    p_fp32 = p_fp32.float()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_fp32)
                    state['previous_grad'] = torch.zeros_like(p_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_fp32)
                    state['previous_grad'] = state['previous_grad'].type_as(p_fp32)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                state['step'] += 1
                beta1, beta2 = group['betas']

                bias_correction1 = 1.0 - beta1 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # compute diffGrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state['previous_grad'] = grad.clone()

                buffered = group['buffer'][state['step'] % 10]
                if state['step'] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma_max = 2.0 / (1.0 - beta2) - 1.0
                    n_sma = n_sma_max - 2.0 * state['step'] * beta2_t / (1.0 - beta2_t)
                    buffered[1] = n_sma

                    if n_sma >= self.n_sma_threshold:
                        rt = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        )

                        step_size = rt
                        if not group['adamd_debias_term']:
                            step_size /= bias_correction1
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / bias_correction1
                    else:
                        step_size = -1

                    buffered[2] = step_size

                if n_sma >= self.n_sma_threshold:
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])

                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    # update momentum with dfc
                    p_fp32.addcdiv_(exp_avg * dfc.float(), de_nom, value=-step_size * group['lr'])
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])

                    p_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                if p.dtype in (torch.float16, torch.bfloat16):
                    p.copy_(p_fp32)

        return loss
