import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaSmooth(BaseOptimizer):
    r"""An Adaptive Learning Rate Method based on Effective Ratio.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.5, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaSmooth'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['prev_param'] = torch.zeros_like(p)
                state['s'] = torch.zeros_like(p)
                state['n'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['prev_param'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)
                    state['n'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                prev_param = state['prev_param']
                p_diff = p - prev_param

                s, n = state['s'], state['n']
                s.add_(p_diff)
                n.add_(p_diff.abs())

                c = s.sum().abs_().div_(n.sum())  # e_t
                c.mul_(beta2 - beta1).add_(1.0 - beta2)

                c_p2 = c.pow(2)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(1.0 - c_p2).addcmul_(grad, grad, value=c_p2)

                step_size = torch.full_like(exp_avg_sq, fill_value=group['lr'])
                step_size.div_((exp_avg_sq + group['eps']).sqrt()).mul_(grad)

                p.add_(-step_size)

                state['prev_param'].copy_(p)

        return loss
