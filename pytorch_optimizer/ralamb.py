import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class RaLamb(Optimizer, BaseOptimizer):
    """
    Reference : https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20
    Example :
        from pytorch_optimizer import RaLamb
        ...
        model = YourModel()
        optimizer = RaLamb(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adamd_debias_term: bool = False,
        pre_norm: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
    ):
        """RaLamb
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param pre_norm: bool. perform pre-normalization of all gradients
        :param n_sma_threshold: int. (recommended is 5)
        :param degenerated_to_sgd: float. degenerated to SGD
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.adamd_debias_term = adamd_debias_term
        self.pre_norm = pre_norm
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

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

    @torch.no_grad()
    def get_gradient_norm(self) -> float:
        norm_sq: float = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                norm_sq += torch.linalg.norm(p.grad).cpu().numpy() ** 2

        norm = math.sqrt(norm_sq)

        return norm

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm: float = 1.0
        if self.pre_norm:
            grad_norm = self.get_gradient_norm()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.pre_norm:
                    p.grad /= grad_norm

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RaLamb does not support sparse gradients')

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
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state['step'] += 1
                buffered = group['buffer'][state['step'] % 10]

                bias_correction1 = 1.0 - beta1 ** state['step']

                if state['step'] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma

                    # more conservative since it's an approximated value
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
                        step_size = group['lr'] / bias_correction1

                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])

                radam_step = p_fp32.clone()
                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(exp_avg, de_nom, value=-step_size)
                else:
                    radam_step.add_(exp_avg, alpha=-step_size)

                radam_step = radam_step.pow(2).sum().sqrt()
                weight_norm = p.pow(2).sum().sqrt().clamp(0.0, self.clamp)
                if weight_norm == 0 or radam_step == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / radam_step

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_step
                state['trust_ratio'] = trust_ratio

                if n_sma >= self.n_sma_threshold:
                    p_fp32.addcdiv_(exp_avg, de_nom, value=-step_size * trust_ratio)
                else:
                    p_fp32.add_(exp_avg, alpha=-step_size * trust_ratio)

                if p.dtype in (torch.float16, torch.bfloat16):
                    p.copy_(p_fp32)

        return loss
