from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import compile_foreach_step, group_tensors_by_device_and_dtype


class FTRL(BaseOptimizer):
    """Follow the Regularized Leader updates with L1 and L2 penalties.

    Compiled foreach updates keep float16 and bfloat16 arithmetic eager to preserve update rounding.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        lr_power: Exponent controlling the accumulated gradient correction, typically `-0.5`.
        beta: Offset in the adaptive update denominator.
        lambda_1: L1 regularization parameter.
        lambda_2: L2 regularization parameter.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations with additional temporary memory. `None` enables them when supported.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        lr_power: float = -0.5,
        beta: float = 0.0,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(beta, 'beta')
        self.validate_non_positive(lr_power, 'lr_power')
        self.validate_non_negative(lambda_1, 'lambda_1')
        self.validate_non_negative(lambda_2, 'lambda_2')

        self.maximize = maximize
        self.foreach = foreach
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'lr_power': lr_power,
            'beta': beta,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FTRL'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

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
                state['z'] = torch.zeros_like(p)
                state['n'] = torch.zeros_like(p)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        zs: list[torch.Tensor],
        ns: list[torch.Tensor],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        sigmas = torch._foreach_pow(grads, 2)
        de_noms = torch._foreach_pow(ns, -group['lr_power'])
        new_ns = torch._foreach_add(ns, sigmas)

        sigmas = torch._foreach_pow(new_ns, -group['lr_power'])
        torch._foreach_sub_(sigmas, de_noms)
        torch._foreach_div_(sigmas, group['lr'])

        new_zs = torch._foreach_add(zs, grads)
        torch._foreach_mul_(sigmas, params)
        torch._foreach_sub_(new_zs, sigmas)

        de_noms = torch._foreach_sqrt(new_ns)
        torch._foreach_add_(de_noms, group['beta'])
        torch._foreach_div_(de_noms, group['lr'])
        torch._foreach_add_(de_noms, group['lambda_2'])

        updates = torch._foreach_sign(new_zs)
        torch._foreach_mul_(updates, group['lambda_1'])
        torch._foreach_sub_(updates, new_zs)
        torch._foreach_div_(updates, de_noms)

        for update, z in zip(updates, new_zs):
            update.masked_fill_(z.abs() <= group['lambda_1'], 0.0)

        torch._foreach_copy_(params, updates)
        torch._foreach_copy_(zs, new_zs)
        torch._foreach_copy_(ns, new_ns)

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
                params, grads, state_dict = self.collect_trainable_params(group, self.state, state_keys=['z', 'n'])
                for batch in group_tensors_by_device_and_dtype(params, grads, state_dict):
                    update = self._step_foreach
                    if self._compiled_step_foreach is not None and batch['params'][0].dtype not in (
                        torch.float16, torch.bfloat16
                    ):
                        update = self._compiled_step_foreach
                    update(group, batch['params'], batch['grads'], batch['z'], batch['n'])
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                z, n = state['z'], state['n']

                p, grad, z, n = self.view_as_real(p, grad, z, n)

                sigma = grad.pow(2)
                de_nom = n.pow(-group['lr_power'])
                n.add_(sigma)

                torch.pow(n, -group['lr_power'], out=sigma)
                sigma.sub_(de_nom).div_(group['lr'])

                z.add_(grad).sub_(sigma.mul_(p))

                torch.sqrt(n, out=de_nom)
                de_nom.add_(group['beta']).div_(group['lr']).add_(group['lambda_2'])

                torch.sign(z, out=p)
                p.mul_(group['lambda_1']).sub_(z).div_(de_nom)
                p.masked_fill_(z.abs() <= group['lambda_1'], 0.0)

        return loss
