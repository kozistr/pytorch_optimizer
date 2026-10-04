import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdaSmooth(BaseOptimizer):
    """Adaptive updates with effective ratio smoothing.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Lower and upper smoothing bounds for the effective ratio adaptation.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.5, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaSmooth'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['prev_param'] = p.clone()
                state['s'] = torch.zeros_like(p)
                state['n'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                s, n, prev_param, exp_avg_sq = state['s'], state['n'], state['prev_param'], state['exp_avg_sq']

                p, grad, s, n, prev_param, exp_avg_sq = self.view_as_real(p, grad, s, n, prev_param, exp_avg_sq)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p_diff = p - prev_param

                s.add_(p_diff)
                n.add_(p_diff.abs())

                c = s.abs().div_(n.add(group['eps']))
                c.mul_(beta2 - beta1).add_(1.0 - beta2)

                c_p2 = c.pow(2)

                exp_avg_sq.lerp_(grad.square(), weight=c_p2)

                step_size = torch.full_like(exp_avg_sq, fill_value=group['lr'])
                step_size.div_((exp_avg_sq + group['eps']).sqrt()).mul_(grad)

                state['prev_param'].copy_(torch.view_as_complex(p) if torch.is_complex(state['prev_param']) else p)

                p.add_(-step_size)

        return loss
