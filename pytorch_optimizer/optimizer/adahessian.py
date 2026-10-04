import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, HutchinsonG, Loss, ParamGroup, ParamsT


class AdaHessian(BaseOptimizer):
    """Adaptive second-order updates using Hutchinson Hessian estimates.

    Use `loss.backward(create_graph=True)` for internal Hessian estimation, or supply
    external estimates through `step(hessian=...)`.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient mean and squared Hessian diagonal estimates.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        hessian_power: Exponent applied to the root mean square Hessian diagonal estimate.
        update_period: Number of steps after which to apply the Hessian approximation.
        num_samples: Number of noise samples for each Hessian diagonal estimate.
        hessian_distribution: Type of distribution used to initialize the Hutchinson trace estimator.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        hessian_power: float = 1.0,
        update_period: int = 1,
        num_samples: int = 1,
        hessian_distribution: HutchinsonG = 'rademacher',
        eps: float = 1e-16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_range(hessian_power, 'Hessian Power', 0, 1, range_type='(]')

        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'hessian_power': hessian_power,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaHessian'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_hessian_diag_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None, hessian: list[torch.Tensor] | None = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)

        step: int = self.param_groups[0]['step'] + 1
        update_hessian = hessian is not None or (step - 1) % self.update_period == 0

        if hessian is not None:
            self.set_hessian(self.param_groups, self.state, hessian)
        elif update_hessian:
            self.zero_hessian(self.param_groups, self.state)
            self.compute_hutchinson_hessian(
                param_groups=self.param_groups,
                state=self.state,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )

        for group in self.param_groups:
            group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = self.apply_adam_debias(group.get('adam_debias', False), group['lr'], bias_correction1)

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

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                exp_avg.lerp_(grad, weight=1.0 - beta1)

                if 'hessian' in state and update_hessian:
                    exp_hessian_diag_sq.mul_(beta2).addcmul_(state['hessian'], state['hessian'], value=1.0 - beta2)

                de_nom = (exp_hessian_diag_sq / bias_correction2).pow_(group['hessian_power'] / 2).add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
