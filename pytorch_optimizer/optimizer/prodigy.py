import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Prodigy(Optimizer, BaseOptimizer):
    r"""An Expeditiously Adaptive Parameter-Free Learner.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. betas.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use AdamW style weight decay.
    :param fixed_decay: bool. fix weight decay.
    :param bias_correction: bool. Turn on Adam's bias correction.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        betas: BETAS = (0.9, 0.999),
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        eps: float = 1e-8,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'step': 0,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Prodigy'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = group['params'][0].device

        beta1, beta2 = group['betas']

        beta2_sq: float = math.sqrt(beta2)

        d: float = group['d']
        lr: float = group['lr']

        bias_correction1: float = 1.0 - beta1 ** (group['step'] + 1)
        bias_correction2_sq: float = math.sqrt(1.0 - beta2 ** (group['step'] + 1))
        bias_correction: float = bias_correction1 / bias_correction2_sq

        # it's not Adam Debias
        d_lr: float = self.apply_adam_debias(
            not group['bias_correction'], step_size=d * lr, bias_correction1=bias_correction
        )

        sk_l1 = torch.tensor([0.0], device=device)
        numerator_acc = torch.tensor([0.0], device=device)

        if 'numerator_weighted' not in group:
            group['numerator_weighted'] = torch.tensor([0.0], device=device)
        numerator_weighted = group['numerator_weighted']

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if 'step' not in state:
                    state['s'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, s = state['exp_avg'], state['exp_avg_sq'], state['s']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                numerator_acc.add_(torch.dot(grad.flatten(), s.div(de_nom).flatten()), alpha=d_lr)

                exp_avg.mul_(beta1).add_(grad, alpha=d_lr * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                s.mul_(beta2_sq).add_(grad, alpha=d_lr * (1.0 - beta2_sq))

                sk_l1.add_(s.abs().sum())

        if sk_l1 == 0:
            return loss

        numerator_weighted.mul_(beta2_sq).add_(numerator_acc, alpha=1.0 - beta2_sq)  # fmt: skip

        if lr > 0.0:
            d_hat = numerator_weighted / (1.0 - beta2_sq) * sk_l1
            d = max(d, min(d_hat.item(), d * group['growth_rate']))

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.addcdiv_(exp_avg, de_nom, value=-1.0)

        return loss
