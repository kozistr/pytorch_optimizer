import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaBelief(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/juntang-zhuang/Adabelief-Optimizer
    Example :
        from pytorch_optimizer import AdaBelief
        ...
        model = YourModel()
        optimizer = AdaBelief(model.parameters())
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
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = True,
        degenerated_to_sgd: bool = True,
        amsgrad: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-16,
    ):
        """AdaBelief optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param n_sma_threshold: (recommended is 5)
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW
        :param fixed_decay: bool. fix weight decay
        :param rectify: bool. perform the rectified update similar to RAdam
        :param degenerated_to_sgd: bool. perform SGD update when variance of gradient is high
        :param amsgrad: bool. whether to use the AMSBound variant
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.n_sma_threshold = n_sma_threshold
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.degenerated_to_sgd = degenerated_to_sgd
        self.adamd_debias_term = adamd_debias_term
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
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
                state['exp_avg_var'] = torch.zeros_like(p)
                if group['amsgrad']:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p)

                if self.weight_decouple:
                    decay: float = (
                        group['lr'] * group['weight_decay'] if not self.fixed_decay else group['weight_decay']
                    )
                    p_fp32.mul_(1.0 - decay)
                elif group['weight_decay'] != 0:
                    grad.add_(p_fp32, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                beta1, beta2 = group['betas']

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

                exp_avg_var = exp_avg_var.add_(group['eps'])
                if group['amsgrad']:
                    exp_avg_var = torch.max(state['max_exp_avg_var'], exp_avg_var)

                de_nom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    step_size = group['lr']
                    if not group['adamd_debias_term']:
                        step_size /= bias_correction1
                    p_fp32.addcdiv_(exp_avg, de_nom, value=-step_size)
                else:
                    buffered = group['buffer'][state['step'] % 10]
                    if state['step'] == buffered[0]:
                        n_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        n_sma_max = 2 / (1 - beta2) - 1
                        n_sma = n_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
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
                        de_nom = exp_avg_var.sqrt().add_(group['eps'])
                        p_fp32.addcdiv_(exp_avg, de_nom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                if p.dtype in (torch.float16, torch.bfloat16):
                    p.copy_(p_fp32)

        return loss
