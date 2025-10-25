import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class AdaMax(BaseOptimizer):
    """An Adaptive and Momental Bound Method for Stochastic Learning.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
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
        return 'AdaMax'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_inf'] = torch.zeros_like(p)

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'],
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']

                p, grad, exp_avg, exp_inf = self.view_as_real(p, grad, exp_avg, exp_inf)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)

                norm_buf = torch.cat(
                    (exp_inf.mul_(beta2).unsqueeze(0), grad.abs().add_(group['eps']).unsqueeze_(0)),
                    dim=0,
                )
                torch.max(norm_buf, dim=0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

                p.addcdiv_(exp_avg, exp_inf, value=-step_size)

        return loss
