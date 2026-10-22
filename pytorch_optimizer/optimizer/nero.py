
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


def channel_view(x: torch.Tensor) -> torch.Tensor:
    """Flatten a tensor to one row per output channel."""
    return x.view(x.size()[0], -1)


def neuron_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute the L2 norm of the weights for each output neuron."""
    if x.dim() <= 1:
        return x.abs()

    view_shape: list[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).norm(dim=1).view(*view_shape)


def neuron_mean(x: torch.Tensor) -> torch.Tensor:
    """Compute the mean of the weights for each output neuron."""
    if x.dim() <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape: list[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).mean(dim=1).view(*view_shape)


class Nero(BaseOptimizer):
    """Neuron wise adaptive updates with optional weight constraints.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        beta: Decay rate for squared neuron wise gradient norms.
        constraints: Center and normalize weights with more than one dimension after each update.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 0.01,
        beta: float = 0.999,
        constraints: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'lr': lr, 'beta': beta, 'constraints': constraints, 'eps': eps, 'foreach': foreach}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Nero'

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

            if len(state) == 0:
                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p).add_(group['eps']))

                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))

                state['scale'] = neuron_norm(p).mean()
                if state['scale'] == 0.0:
                    state['scale'] = 0.01

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        scales: list[torch.Tensor],
        bias_correction: float | torch.Tensor,
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        norms = [neuron_norm(grad) for grad in grads]
        torch._foreach_mul_(exp_avg_sqs, group['beta'])
        torch._foreach_addcmul_(exp_avg_sqs, norms, norms, value=1.0 - group['beta'])

        de_noms = torch._foreach_div(exp_avg_sqs, bias_correction)
        torch._foreach_sqrt_(de_noms)
        torch._foreach_add_(de_noms, group['eps'])
        updates = torch._foreach_div(grads, de_noms)
        for update in updates:
            update.nan_to_num_(nan=0.0)
        torch._foreach_mul_(updates, scales)
        foreach_add_(params, updates, alpha=-group['lr'])

        if group['constraints']:
            constrained = [p for p in params if p.dim() > 1]
            if constrained:
                torch._foreach_sub_(constrained, [neuron_mean(p) for p in constrained])
                norms = [neuron_norm(p) for p in constrained]
                torch._foreach_add_(norms, group['eps'])
                torch._foreach_div_(constrained, norms)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            bias_correction: float = self.debias(group['beta'], group['step'])

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=['exp_avg_sq'])
                state['scale'] = [torch.as_tensor(self.state[p]['scale'], device=p.device) for p in params]
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(
                        group, batch['params'], batch['grads'], batch['exp_avg_sq'], batch['scale'], bias_correction
                    )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_norm = neuron_norm(grad)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(group['beta']).addcmul_(grad_norm, grad_norm, value=1.0 - group['beta'])

                grad_normed = grad / ((exp_avg_sq / bias_correction).sqrt_().add_(group['eps']))
                torch.nan_to_num(grad_normed, nan=0.0, out=grad_normed)

                p.add_(grad_normed, alpha=-group['lr'] * state['scale'])

                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p).add_(group['eps']))

        return loss
