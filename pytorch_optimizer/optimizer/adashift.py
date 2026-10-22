from collections import deque
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class AdaShift(BaseOptimizer):
    """Adaptive updates with temporally decorrelated gradient moments.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        keep_num: Number of gradients used to compute first moment estimation.
        reduce_func: Function applied to squared gradients to reduce correlation. If None, no function is applied.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.999),
        keep_num: int = 10,
        reduce_func: Callable | None = torch.max,
        eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(keep_num, 'keep_num')
        self.validate_non_negative(eps, 'eps')

        self.reduce_func: Callable = reduce_func if reduce_func is not None else lambda x: x
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'lr': lr, 'betas': betas, 'keep_num': keep_num, 'eps': eps, 'foreach': foreach, **kwargs}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaShift'

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
                state['grad_queue'] = deque(maxlen=group['keep_num'])
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        offsets: list[torch.Tensor],
        bias_correction: float | torch.Tensor,
        first_grad_weight: float | torch.Tensor,
        last_grad_weight: float | torch.Tensor,
    ) -> None:
        beta1, beta2 = group['betas']
        exp_avgs = state['exp_avg']
        if offsets:
            foreach_add_(exp_avgs, offsets, alpha=-first_grad_weight)
        torch._foreach_mul_(exp_avgs, beta1)
        foreach_add_(exp_avgs, grads, alpha=last_grad_weight)

        if not offsets:
            return

        reduced = [self.reduce_func(offset.square()) for offset in offsets]
        torch._foreach_lerp_(state['exp_avg_sq'], reduced, weight=1.0 - beta2)
        de_noms = torch._foreach_div(state['exp_avg_sq'], bias_correction)
        torch._foreach_sqrt_(de_noms)
        torch._foreach_add_(de_noms, group['eps'])
        updates = [exp_avg.clone() for exp_avg in exp_avgs] if group.get('cautious') else exp_avgs
        if group.get('cautious'):
            for update, grad in zip(updates, grads):
                self.apply_cautious(update, grad)
        updates = torch._foreach_div(updates, de_noms)
        foreach_add_(params, updates, alpha=-group['lr'])

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

            exp_weight_sum: int = sum(beta1 ** i for i in range(group['keep_num']))  # fmt: skip
            first_grad_weight: float = beta1 ** (group['keep_num'] - 1) / exp_weight_sum
            last_grad_weight: float = 1.0 / exp_weight_sum

            bias_correction: float = self.debias(beta2, max(1, group['step'] - group['keep_num']))

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                offsets = []
                for p, grad in zip(params, grads):
                    self.maximize_gradient(grad, maximize=self.maximize)
                    queue = self.state[p]['grad_queue']
                    offsets.append(queue[0] if len(queue) == group['keep_num'] else None)
                    queue.append(grad.clone())
                for ready in (False, True):
                    indices = [i for i, offset in enumerate(offsets) if (offset is not None) == ready]
                    selected = [params[i] for i in indices]
                    selected_grads = [grads[i] for i in indices]
                    state = {key: [self.state[p][key] for p in selected] for key in ['exp_avg', 'exp_avg_sq']}
                    if ready:
                        state['offsets'] = [offsets[i] for i in indices]
                    for batch in group_tensors_by_device_and_dtype(selected, selected_grads, state):
                        self._step_foreach(
                            group, batch['params'], batch['grads'], batch, batch.get('offsets', []),
                            bias_correction, first_grad_weight, last_grad_weight,
                        )
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_queue = state['grad_queue']
                offset_grad = grad_queue[0] if len(grad_queue) == group['keep_num'] else None
                grad_queue.append(grad.clone())

                exp_avg = state['exp_avg']
                if offset_grad is None:
                    exp_avg.mul_(beta1).add_(grad, alpha=last_grad_weight)
                    continue

                exp_avg.sub_(offset_grad, alpha=first_grad_weight).mul_(beta1).add_(grad, alpha=last_grad_weight)

                reduced_grad_sq = self.reduce_func(offset_grad.square())

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.lerp_(reduced_grad_sq, weight=1.0 - beta2)

                if group.get('cautious'):
                    update = exp_avg.clone()
                    self.apply_cautious(update, grad)
                    update.div_(exp_avg_sq.div(bias_correction).sqrt_().add_(group['eps']))
                else:
                    update = exp_avg.div(exp_avg_sq.div(bias_correction).sqrt_().add_(group['eps']))

                p.add_(update, alpha=-group['lr'])

        return loss
