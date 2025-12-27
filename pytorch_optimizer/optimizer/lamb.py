from typing import List, Optional, Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Lamb(BaseOptimizer):
    """Large Batch Optimization for Deep Learning.

    This Lamb implementation is based on the paper v3, which does not use de-biasing.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        rectify (bool): Perform the rectified update similar to RAdam.
        degenerated_to_sgd (bool): Degenerate to SGD.
        n_sma_threshold (int): Recommended is 5.
        grad_averaging (bool): Whether to apply (1 - beta2) to gradient when calculating running averages of gradient.
        max_grad_norm (float): Max gradient norm to clip.
        adam (bool): Always use trust ratio = 1, which turns this into Adam. Useful for comparison purposes.
        pre_norm (bool): Perform pre-normalization of all gradients.
        eps (float): Term added to the denominator to improve numerical stability.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: Parameters,
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
        foreach: Optional[bool] = None,
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
        """Check if foreach can be used for this group.

        Foreach is disabled when using features that require per-parameter handling:
        - AdaNorm
        - Rectify (has conditional logic per parameter)
        """
        if group.get('foreach') is False:
            return False

        if group.get('adanorm') or group.get('rectify'):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        grad_norm: Union[torch.Tensor, float],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        step_size: float,
    ) -> None:
        beta1, beta2 = group['betas']
        eps = group['eps']
        beta3: float = 1.0 - beta1 if group['grad_averaging'] else 1.0

        if self.maximize:
            torch._foreach_neg_(grads)

        if self.pre_norm:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

            torch._foreach_div_(grads, grad_norm)

        self.apply_weight_decay_foreach(
            params=params,
            grads=grads,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
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

        weight_norms = torch._foreach_norm(params)
        torch._foreach_clamp_max_(weight_norms, self.clamp)

        p_norms = torch._foreach_norm(updates)

        for p, update, wn, pn in zip(params, updates, weight_norms, p_norms):
            trust_ratio: float = 1.0
            if wn != 0 and pn != 0:
                trust_ratio = (wn / (pn + eps)).item()

            state = self.state[p]
            state['weight_norm'] = wn
            state['adam_norm'] = pn
            state['trust_ratio'] = trust_ratio

            if group['adam']:
                trust_ratio = 1.0

            p.add_(update, alpha=-step_size * trust_ratio)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    def update(
        self,
        p: torch.Tensor,
        group: ParamGroup,
        grad_norm: Union[torch.Tensor, float],
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
            grad.div_(grad_norm)

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

        de_nom: Optional[torch.Tensor] = None

        if group['rectify']:
            update = p.clone()
            if n_sma >= self.n_sma_threshold:
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                update.addcdiv_(exp_avg, de_nom, value=-step_size)
            else:
                update.add_(exp_avg, alpha=-step_size)
        else:
            update = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])

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
