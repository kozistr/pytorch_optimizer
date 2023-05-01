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
    :param degenerated_to_sgd: bool. perform SGD update when variance of gradient is high.
    :param use_gc: bool. use Gradient Centralization (both convolution & fc layers).
    :param gc_conv_only: bool. use Gradient Centralization (only convolution layer).
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        betas: BETAS = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        use_gc: bool = True,
        gc_conv_only: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
    ):
        self.lr = lr
        self.alpha = alpha
        self.k = k
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
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
            'adanorm': adanorm,
            'adam_debias': adam_debias,
            'eps': eps,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_lookahead_k(self.k)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'Ranger'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['slow_buffer'] = torch.empty_like(p)
                state['slow_buffer'].copy_(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']
            bias_correction1: float = 1.0 - beta1 ** group['step']

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=True,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                bias_correction1=bias_correction1,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
                adam_debias=group['adam_debias'],
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['slow_buffer'] = torch.empty_like(p)
                    state['slow_buffer'].copy_(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                if self.use_gc and grad.dim() > self.gc_gradient_threshold:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                s_grad = grad
                if group['adanorm']:
                    grad_norm = torch.linalg.norm(grad)

                    exp_grad_norm = state['exp_grad_norm']
                    exp_grad_norm.mul_(group['r']).add_(grad_norm, alpha=1.0 - group['r'])

                    if exp_grad_norm > grad_norm:
                        s_grad *= exp_grad_norm / grad_norm

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                else:
                    p.add_(exp_avg, alpha=-step_size)

                if group['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p - slow_p, alpha=self.alpha)
                    p.copy_(slow_p)

        return loss
