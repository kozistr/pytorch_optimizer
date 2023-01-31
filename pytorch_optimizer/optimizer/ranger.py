import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient


class Ranger(Optimizer, BaseOptimizer):
    r"""a synergistic optimizer combining RAdam and LookAhead, and now GC in one optimizer.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param n_sma_threshold: int. (recommended is 5).
    :param use_gc: bool. use Gradient Centralization (both convolution & fc layers).
    :param gc_conv_only: bool. use Gradient Centralization (only convolution layer).
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshold: int = 5,
        betas: BETAS = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        use_gc: bool = True,
        gc_conv_only: bool = False,
        adamd_debias_term: bool = False,
    ):
        self.lr = lr
        self.alpha = alpha
        self.k = k
        self.n_sma_threshold = n_sma_threshold
        self.betas = betas
        self.weight_decay = weight_decay
        self.use_gc = use_gc
        self.eps = eps

        self.gc_gradient_threshold: int = 3 if gc_conv_only else 1

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'k': k,
            'step_counter': 0,
            'n_sma_threshold': n_sma_threshold,
            'weight_decay': weight_decay,
            'adamd_debias_term': adamd_debias_term,
            'buffer': [[None, None, None] for _ in range(10)],
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_lookahead_k(self.k)
        self.validate_epsilon(self.eps)

    @property
    def __str__(self) -> str:
        return 'Ranger'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['slow_buffer'] = torch.empty_like(p)
                state['slow_buffer'].copy_(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            n_sma_max: float = 2.0 / (1.0 - beta2) - 1.0
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__str__)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['slow_buffer'] = torch.empty_like(p)
                    state['slow_buffer'].copy_(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_gc and grad.dim() > self.gc_gradient_threshold:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state['step']

                buffered = group['buffer'][state['step'] % 10]
                if state['step'] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma = n_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma
                    if n_sma > self.n_sma_threshold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        )
                        if not group['adamd_debias_term']:
                            step_size /= bias_correction1
                    else:
                        step_size = 1.0 / bias_correction1

                    buffered[2] = step_size

                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                if n_sma > self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(exp_avg, de_nom, value=-step_size * group['lr'])
                else:
                    p.add_(exp_avg, alpha=-step_size * group['lr'])

                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p - slow_p, alpha=self.alpha)
                    p.copy_(slow_p)

        return loss
