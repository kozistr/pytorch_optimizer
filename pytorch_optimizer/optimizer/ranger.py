import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient


class Ranger(BaseOptimizer):
    """RAdam with Lookahead and optional gradient centralization.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        alpha: Lookahead interpolation factor from slow weights toward fast weights.
        k: Number of steps between Lookahead updates.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        n_sma_threshold: Minimum effective simple moving average length for rectification.
        degenerated_to_sgd: Use an SGD update before the moving average reaches the rectification threshold.
        use_gc: Use Gradient Centralization (both convolution & fc layers).
        gc_conv_only: Use Gradient Centralization (only convolution layer).
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.95, 0.999),
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        use_gc: bool = True,
        gc_conv_only: bool = False,
        eps: float = 1e-5,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(alpha, 'alpha', 0.0, 1.0, range_type='[]')
        self.validate_positive(k, 'k')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.use_gc = use_gc
        self.gc_gradient_threshold: int = 3 if gc_conv_only else 1
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'k': k,
            'step_counter': 0,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Ranger'

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
                state['slow_buffer'] = p.clone()

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

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

            bias_correction1: float = self.debias(beta1, group['step'])

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=True,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, slow_buffer = state['exp_avg'], state['exp_avg_sq'], state['slow_buffer']

                p, grad, exp_avg, exp_avg_sq, slow_buffer = self.view_as_real(
                    p, grad, exp_avg, exp_avg_sq, slow_buffer
                )

                if self.use_gc and grad.dim() > self.gc_gradient_threshold:
                    centralize_gradient(grad, gc_conv_only=False)

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

                exp_avg.lerp_(s_grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                else:
                    p.add_(exp_avg, alpha=-step_size)

                if group['step'] % group['k'] == 0:
                    slow_buffer.lerp_(p, weight=group['alpha'])
                    p.copy_(slow_buffer)

        return loss
