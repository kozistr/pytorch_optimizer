import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdaBound(BaseOptimizer):
    """Adam updates with learning rate bounds that converge to SGD.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        final_lr: Final learning rate.
        betas: Decay rates for the first and second moments.
        gamma: Convergence speed of the bound functions.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        final_lr: float = 1e-1,
        betas: Betas = (0.9, 0.999),
        gamma: float = 1e-3,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'final_lr': final_lr,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
        }

        super().__init__(params, defaults)

        self.base_lrs: list[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'AdaBound'

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
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            self.init_group(group)
            group['step'] += 1

            base_lr = self.base_lrs[group_index]
            if base_lr == 0.0 and group['lr'] > 0.0:
                base_lr = self.base_lrs[group_index] = group['lr']

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            final_lr: float = group['final_lr'] * group['lr'] / base_lr if base_lr > 0.0 else 0.0
            lower_bound: float = final_lr * (1 - 1 / (group['gamma'] * group['step'] + 1))
            upper_bound: float = final_lr * (1 + 1 / (group['gamma'] * group['step']))

            step_size = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'] * bias_correction2_sq,
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )

                update = torch.full_like(de_nom, fill_value=step_size)
                update.div_(de_nom).clamp_(min=lower_bound, max=upper_bound).mul_(exp_avg)

                p.add_(-update)

        return loss
