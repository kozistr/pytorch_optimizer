
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Lamb(BaseOptimizer):
    """Adam updates with a trust ratio for each parameter tensor.

    The default update follows version 3 of the paper without bias correction.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        rectify: Perform the rectified update similar to RAdam.
        degenerated_to_sgd: Use an SGD update before the moving average reaches the rectification threshold.
        n_sma_threshold: Minimum effective simple moving average length for rectification.
        grad_averaging: Scale new gradient contributions by `1 - beta1`.
        max_grad_norm: Reference norm for gradient scaling when `pre_norm=True`. `0` disables scaling.
        adam: Use a trust ratio of 1 for all parameters.
        pre_norm: Divide gradients by a scaling factor derived from their global norm.
        eps: Term added to the denominator to improve numerical stability.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
        maximize: Maximize the objective instead of minimizing it.

    """

    clamp: float = 10.0

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        degenerated_to_sgd: bool = False,
        n_sma_threshold: int = 5,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        pre_norm: bool = False,
        eps: float = 1e-6,
        foreach: bool | None = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.degenerated_to_sgd = degenerated_to_sgd
        self.n_sma_threshold = n_sma_threshold
        self.pre_norm = pre_norm
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'grad_averaging': grad_averaging,
            'max_grad_norm': max_grad_norm,
            'adam': adam,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lamb'

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

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        """Check tensor compatibility and options for batched updates.

        Disable batched updates when using AdaNorm or rectification.
        """
        if group.get('foreach') is False:
            return False

        if group.get('adanorm') or group.get('rectify'):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        grad_norm: torch.Tensor | float,
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        step_size: float,
    ) -> None:
        beta1, beta2 = group['betas']
        eps = group['eps']
        beta3: float = 1.0 - beta1 if group['grad_averaging'] else 1.0

        if self.maximize:
            torch._foreach_neg_(grads)

        if self.pre_norm:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.reshape(())

            torch._foreach_mul_(grads, grad_norm)

        if group['weight_decouple']:
            self.apply_weight_decay_foreach(
                params=params,
                grads=grads,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=True,
                fixed_decay=group['fixed_decay'],
            )

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=beta3)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        updates = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(updates, eps)
        torch._foreach_reciprocal_(updates)
        torch._foreach_mul_(updates, exp_avgs)

        if not group['weight_decouple'] and group['weight_decay'] > 0.0:
            torch._foreach_add_(updates, params, alpha=group['weight_decay'])

        weight_norms = torch._foreach_norm(params)
        torch._foreach_clamp_max_(weight_norms, self.clamp)

        p_norms = torch._foreach_norm(updates)

        trust_ratios = torch._foreach_div(weight_norms, torch._foreach_add(p_norms, eps))
        trust_ratios = [
            torch.where((wn != 0) & (pn != 0), ratio, torch.ones_like(ratio))
            for wn, pn, ratio in zip(weight_norms, p_norms, trust_ratios)
        ]

        for p, wn, pn, trust_ratio in zip(params, weight_norms, p_norms, trust_ratios):
            state = self.state[p]
            state['weight_norm'] = wn
            state['adam_norm'] = pn
            state['trust_ratio'] = trust_ratio

        if not group['adam']:
            torch._foreach_mul_(updates, trust_ratios)

        torch._foreach_add_(params, updates, alpha=-step_size)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> torch.Tensor | float:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    def update(
        self,
        p: torch.Tensor,
        group: ParamGroup,
        grad_norm: torch.Tensor | float,
        n_sma: float,
        step_size: float,
        beta1: float,
        beta2: float,
        beta3: float,
    ) -> None:
        grad = p.grad
        if grad is None:
            return

        if self.pre_norm:
            grad.mul_(grad_norm)

        self.maximize_gradient(grad, maximize=self.maximize)

        state = self.state[p]

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

        s_grad = self.get_adanorm_gradient(
            grad=grad,
            adanorm=group.get('adanorm', False),
            exp_grad_norm=state.get('exp_grad_adanorm', None),
            r=group.get('adanorm_r', None),
        )

        exp_avg.mul_(beta1).add_(s_grad, alpha=beta3)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        self.apply_weight_decay(
            p=p,
            grad=None,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        if group['rectify'] and step_size <= 0:
            return

        de_nom: torch.Tensor | None = None

        if group['rectify']:
            update = p.clone()
            if n_sma >= self.n_sma_threshold:
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                update.addcdiv_(exp_avg, de_nom, value=-step_size)
            else:
                update.add_(exp_avg, alpha=-step_size)
        else:
            update = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])
            if not group['weight_decouple'] and group['weight_decay'] > 0.0:
                update.add_(p, alpha=group['weight_decay'])

        weight_norm = torch.linalg.norm(p).clamp_(min=0, max=self.clamp)
        p_norm = torch.linalg.norm(update)
        trust_ratio: float = 1.0 if weight_norm == 0 or p_norm == 0 else weight_norm / (p_norm + group['eps'])

        state['weight_norm'] = weight_norm
        state['adam_norm'] = p_norm
        state['trust_ratio'] = trust_ratio

        if group['adam']:
            trust_ratio = 1.0

        if group['rectify']:
            if n_sma >= self.n_sma_threshold:
                p.addcdiv_(exp_avg, de_nom, value=-step_size * trust_ratio)
            else:
                p.add_(exp_avg, alpha=-step_size * trust_ratio)
        else:
            p.add_(update, alpha=-step_size * trust_ratio)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm = 1.0
        if self.pre_norm:
            grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            beta3: float = 1.0 - beta1 if group['grad_averaging'] else 1.0
            bias_correction1: float = self.debias(beta1, group['step'])

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group['rectify'],
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

            if self._can_use_foreach(group):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                if params:
                    self._step_foreach(
                        group, params, grads, grad_norm, state_dict['exp_avg'], state_dict['exp_avg_sq'], step_size
                    )
            else:
                for p in group['params']:
                    self.update(p, group, grad_norm, n_sma, step_size, beta1, beta2, beta3)

        return loss
