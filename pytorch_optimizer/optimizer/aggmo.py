import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class AggMo(BaseOptimizer):
    """SGD with aggregated momentum buffers at multiple decay rates.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the momentum buffers to aggregate.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
            CPU float16 and bfloat16 parameters use per-parameter updates.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.0, 0.9, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AggMo'

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
                state['momentum_buffer'] = {beta: torch.zeros_like(p) for beta in group['betas']}

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        # CPU foreach multiplication rounds scalar factors to the buffer dtype.
        return self.can_use_foreach(group, group.get('foreach', self.foreach)) and all(
            p.dtype not in (torch.float16, torch.bfloat16) or p.device.type != 'cpu'
            for p in group['params'] if p.grad is not None
        )

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        momentum_buffers: list[list[torch.Tensor]],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        self.apply_weight_decay_foreach(
            params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        step_size = -group['lr'] / len(group['betas'])
        for beta, buffers in zip(group['betas'], momentum_buffers):
            torch._foreach_mul_(buffers, beta)
            torch._foreach_add_(buffers, grads)

            foreach_add_(params, buffers, alpha=step_size)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            betas = group['betas']
            step_size = -group['lr'] / len(betas)

            if self._can_use_foreach(group):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                for batch in group_tensors_by_device_and_dtype(params, grads):
                    momentum_buffers = [
                        [self.state[p]['momentum_buffer'][beta] for p in batch['params']] for beta in betas
                    ]
                    self._step_foreach(group, batch['params'], batch['grads'], momentum_buffers)
                continue

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

                for beta in betas:
                    buf = state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(grad)

                    if isinstance(step_size, torch.Tensor):
                        p.add_(buf * step_size)
                    else:
                        p.add_(buf, alpha=step_size)

        return loss
