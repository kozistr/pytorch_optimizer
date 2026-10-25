
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcdiv_, group_tensors_by_device_and_dtype


class QHAdam(BaseOptimizer):
    """Adam with quasi-hyperbolic moment averaging.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient mean and squared gradients.
        nus: Weights of the running averages relative to the current gradient and its square.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations with additional temporary memory. `None` enables them when supported.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        nus: tuple[float, float] = (1.0, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_nus(nus)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'nus': nus,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'QHAdam'

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
                state['beta1_weight'] = torch.zeros((1,), dtype=torch.float32, device=grad.device)
                state['beta2_weight'] = torch.zeros((1,), dtype=torch.float32, device=grad.device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        beta1_weights: list[torch.Tensor],
        beta2_weights: list[torch.Tensor],
    ) -> None:
        beta1, beta2 = group['betas']
        nu1, nu2 = group['nus']

        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        torch._foreach_mul_(beta1_weights, beta1)
        torch._foreach_add_(beta1_weights, 1.0)
        torch._foreach_mul_(beta2_weights, beta2)
        torch._foreach_add_(beta2_weights, 1.0)

        weights1 = [weight.to(dtype=grads[0].dtype).reshape(()) for weight in torch._foreach_reciprocal(beta1_weights)]
        weights2 = [weight.to(dtype=grads[0].dtype).reshape(()) for weight in torch._foreach_reciprocal(beta2_weights)]

        for exp_avg, grad, weight in zip(exp_avgs, grads, weights1):
            exp_avg.lerp_(grad, weight=weight)

        grad_p2 = torch._foreach_pow(grads, 2)
        for exp_avg_sq, grad_sq, weight in zip(exp_avg_sqs, grad_p2, weights2):
            exp_avg_sq.lerp_(grad_sq, weight=weight)

        avg_grad_rms = torch._foreach_mul(exp_avg_sqs, nu2)
        if nu2 != 1.0:
            torch._foreach_add_(avg_grad_rms, grad_p2, alpha=1.0 - nu2)

        torch._foreach_sqrt_(avg_grad_rms)
        torch._foreach_add_(avg_grad_rms, group['eps'])

        avg_grads = exp_avgs
        if nu1 != 1.0:
            avg_grads = torch._foreach_mul(exp_avgs, nu1)
            torch._foreach_add_(avg_grads, grads, alpha=1.0 - nu1)

        foreach_addcdiv_(params, avg_grads, avg_grad_rms, value=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq', 'beta1_weight', 'beta2_weight']
                )
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch['exp_avg'], batch['exp_avg_sq'],
                        batch['beta1_weight'], batch['beta2_weight'],
                    )
                continue

            beta1, beta2 = group['betas']
            nu1, nu2 = group['nus']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                beta1_weight, beta2_weight = state['beta1_weight'], state['beta2_weight']
                beta1_weight.mul_(beta1).add_(1.0)
                beta2_weight.mul_(beta2).add_(1.0)

                grad_p2 = grad.pow(2)

                exp_avg.lerp_(grad, weight=beta1_weight.reciprocal().to(dtype=grad.dtype).reshape(()))
                exp_avg_sq.lerp_(grad_p2, weight=beta2_weight.reciprocal().to(dtype=grad.dtype).reshape(()))

                avg_grad = exp_avg
                if nu1 != 1.0:
                    avg_grad = exp_avg.mul(nu1).add_(grad, alpha=1.0 - nu1)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(grad_p2, alpha=1.0 - nu2)

                avg_grad_rms.sqrt_().add_(group['eps'])

                p.addcdiv_(avg_grad, avg_grad_rms, value=-group['lr'])

        return loss
