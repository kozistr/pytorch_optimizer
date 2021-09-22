import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.types import (
    BETAS,
    CLOSURE,
    DEFAULT_PARAMETERS,
    LOSS,
    PARAMS,
    STATE,
)


class AdaBelief(Optimizer):
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
        params: PARAMS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        n_sma_threshold: int = 5,
        amsgrad: bool = False,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = True,
        degenerated_to_sgd: bool = True,
    ):
        """AdaBelief optimizer
        :param params: PARAMS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param n_sma_threshold: (recommended is 5)
        :param amsgrad: bool. whether to use the AMSBound variant
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW
        :param fixed_decay: bool.
        :param rectify: bool. perform the rectified update similar to RAdam
        :param degenerated_to_sgd: bool. perform SGD update when variance of gradient is high
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        self.degenerated_to_sgd = degenerated_to_sgd

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0]
                    or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults: DEFAULT_PARAMETERS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: STATE):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_var'] = torch.zeros_like(p.data)
                if amsgrad:
                    state['max_exp_avg_var'] = torch.zeros_like(p.data)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                half_precision: bool = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients'
                    )

                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']

                    torch.max(
                        max_exp_avg_var,
                        exp_avg_var.add_(group['eps']),
                        out=max_exp_avg_var,
                    )

                    denom = (
                        max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)
                    ).add_(group['eps'])
                else:
                    denom = (
                        exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                if not self.rectify:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        n_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        n_sma_max = 2 / (1 - beta2) - 1
                        n_sma = n_sma_max - 2 * state['step'] * beta2_t / (
                            1 - beta2_t
                        )
                        buffered[1] = n_sma

                        if n_sma >= self.n_sma_threshold:
                            step_size = math.sqrt(
                                (1 - beta2_t)
                                * (n_sma - 4)
                                / (n_sma_max - 4)
                                * (n_sma - 2)
                                / n_sma
                                * n_sma_max
                                / (n_sma_max - 2)
                            ) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if n_sma >= self.n_sma_threshold:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(
                            exp_avg, denom, value=-step_size * group['lr']
                        )
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss
