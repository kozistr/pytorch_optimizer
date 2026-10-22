import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcmul_, group_tensors_by_device_and_dtype


class MSVAG(BaseOptimizer):
    """Adaptive momentum updates with gradient variance damping.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        beta: Moving average (momentum) constant (scalar tensor or float value).
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-2,
        beta: float = 0.9,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'lr': lr, 'beta': beta, 'foreach': foreach}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MSVAG'

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

    @staticmethod
    def get_rho(beta_power: float, beta: float) -> float:
        """Compute the finite step variance correction for a moving average."""
        rho: float = (1.0 - beta_power ** 2) * (1.0 - beta) ** 2  # fmt: skip
        rho /= (1.0 - beta ** 2) * (1.0 - beta_power) ** 2  # fmt: skip
        return min(rho, 0.9999)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        beta_power: float | torch.Tensor,
        rho: float | torch.Tensor,
    ) -> None:
        beta = group['beta']
        exp_avgs, exp_avg_sqs = state['exp_avg'], state['exp_avg_sq']

        if self.maximize:
            torch._foreach_neg_(grads)

        torch._foreach_lerp_(exp_avgs, grads, weight=1.0 - beta)
        torch._foreach_mul_(exp_avg_sqs, beta)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta)

        ms = torch._foreach_div(exp_avgs, 1.0 - beta_power)
        variances = torch._foreach_div(exp_avg_sqs, 1.0 - beta_power)
        mean_sqs = torch._foreach_pow(ms, 2)
        torch._foreach_sub_(variances, mean_sqs)
        torch._foreach_div_(variances, 1.0 - rho)
        torch._foreach_mul_(variances, rho)
        torch._foreach_add_(variances, mean_sqs)

        factors = mean_sqs
        torch._foreach_div_(factors, variances)
        for factor in factors:
            factor.nan_to_num_(nan=0.0)
        torch._foreach_clamp_min_(factors, 0.0)
        torch._foreach_clamp_max_(factors, 1.0)
        foreach_addcmul_(params, ms, factors, value=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta: float = group['beta']
            beta_power: float = beta ** group['step']

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(
                    group, self.state, state_keys=['exp_avg', 'exp_avg_sq']
                )
                rho = self.get_rho(beta_power, beta)
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch, beta_power, rho)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                exp_avg.lerp_(grad, weight=1.0 - beta)
                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                m = exp_avg.div(1.0 - beta_power)
                v = exp_avg_sq.div(1.0 - beta_power)

                rho: float = self.get_rho(beta_power, beta)

                m_p2 = m.pow(2)
                s = v.sub_(m_p2).div_(1.0 - rho)

                factor = m_p2.div_(s.mul_(rho).add_(m_p2))
                torch.nan_to_num(factor, nan=0.0, out=factor)
                factor.clamp_(0.0, 1.0)

                p.addcmul_(m, factor, value=-group['lr'])

        return loss
