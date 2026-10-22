
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_addcdiv_, group_tensors_by_device_and_dtype
from pytorch_optimizer.optimizer.utils.precision import copy_stochastic


class ROSE(BaseOptimizer):
    """Gradient updates scaled by row and column ranges.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        wd_schedule: Scale decoupled decay by `lr / lr_ref`. A float supplies `lr_ref`. `True` reads `max_lr` or
            `initial_lr` from the group. `False` uses `lr`.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        centralize: Subtract the mean of each gradient slice for parameters with two or more dimensions.
        stabilize: Blend local range scaling with the global mean range using coefficient of variation gating.
        bf16_sr: Use stochastic rounding for bfloat16 parameter updates.
        compute_dtype: Data type for gradient and parameter update arithmetic.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        weight_decay: float = 1e-4,
        wd_schedule: bool | float = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool = True,
        compute_dtype: torch.dtype = torch.float64,
        maximize: bool = False,
        foreach: bool | None = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(f'bf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.')

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'weight_decay': weight_decay,
            'wd_schedule': wd_schedule,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'centralize': centralize,
            'stabilize': stabilize,
            'bf16_sr': bf16_sr,
            'compute_dtype': compute_dtype,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ROSE'

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

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        wd_lr: float | torch.Tensor,
    ) -> list[torch.Tensor]:
        use_bf16_sr = group['bf16_sr'] and params[0].dtype == torch.bfloat16
        dtype = torch.float32 if use_bf16_sr and not group['compute_dtype'] else group['compute_dtype']
        originals = grads
        grads = [grad.to(dtype=dtype) for grad in grads]
        values = [p.to(dtype=dtype) for p in params]
        if self.maximize:
            torch._foreach_neg_(grads)
        self.apply_weight_decay_foreach(
            values, grads, wd_lr, group['weight_decay'], group['weight_decouple'], group['fixed_decay']
        )

        updates, de_noms = [], []
        for grad, original in zip(grads, originals):
            if grad.ndim == 0:
                updates.append(grad.sign())
                de_noms.append(torch.ones_like(grad))
                continue
            if grad.ndim == 1:
                g_min, g_max = grad.aminmax()
                de_nom = g_max.abs_().sub_(g_min)
            else:
                axes = tuple(range(1, grad.ndim))
                if group['centralize']:
                    mean = grad.mean(dim=axes, keepdim=True)
                    grad = grad.sub_(mean) if grad is not original else grad.sub(mean)
                raw_scale = grad.amax(dim=axes, keepdim=True).abs_().sub_(grad.amin(dim=axes, keepdim=True))
                if group['stabilize']:
                    std, mean = torch.std_mean(raw_scale, correction=0)
                    trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                    de_nom = mean.lerp(raw_scale, trust)
                else:
                    de_nom = raw_scale
            de_nom.masked_fill_(de_nom == 0.0, 1.0)
            updates.append(grad)
            de_noms.append(de_nom)
        foreach_addcdiv_(values, updates, de_noms, value=-group['lr'])
        if not use_bf16_sr and values[0] is not params[0]:
            torch._foreach_copy_(params, values)
        return values

    def _step_per_param(self, group: ParamGroup, wd_lr: float | torch.Tensor) -> None:
        lr, weight_decay = group['lr'], group['weight_decay']
        compute_dtype = group['compute_dtype']
        for p in group['params']:
            if p.grad is None:
                continue

            use_bf16_sr = group['bf16_sr'] and p.dtype is torch.bfloat16
            fp32 = use_bf16_sr and not compute_dtype

            grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
            param = p.to(dtype=torch.float32 if fp32 else compute_dtype)

            self.maximize_gradient(grad, maximize=self.maximize)

            self.apply_weight_decay(
                param,
                grad,
                lr=wd_lr,
                weight_decay=weight_decay,
                weight_decouple=group['weight_decouple'],
                fixed_decay=group['fixed_decay'],
            )

            if grad.ndim == 0:
                param.add_(grad.sign(), alpha=-lr)
            elif grad.ndim == 1:
                g_min, g_max = grad.aminmax()
                de_nom = g_max.abs_().sub_(g_min)

                de_nom.masked_fill_(de_nom == 0.0, 1.0)
                param.addcdiv_(grad, de_nom, value=-lr)
            else:
                active_axes = tuple(range(1, grad.ndim))

                if group['centralize']:
                    if grad is not p.grad:
                        grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                    else:
                        grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))

                raw_scale = (
                    grad.amax(dim=active_axes, keepdim=True).abs_().sub_(grad.amin(dim=active_axes, keepdim=True))
                )

                if group['stabilize']:
                    std, mean = torch.std_mean(raw_scale, correction=0)

                    trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))

                    de_nom = mean.lerp(raw_scale, trust)
                else:
                    de_nom = raw_scale

                de_nom.masked_fill_(de_nom == 0.0, 1.0)
                param.addcdiv_(grad, de_nom, value=-lr)

            if use_bf16_sr:
                param = param.to(dtype=torch.float32)

                copy_stochastic(p, param)
            elif param is not p:
                p.copy_(param)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            lr = group['lr']
            weight_decay, wd_schedule = group['weight_decay'], group['wd_schedule']

            if weight_decay and wd_schedule:
                wd_lr = lr / (
                    wd_schedule if isinstance(wd_schedule, float) else group.get('max_lr', group.get('initial_lr'))
                )
            else:
                wd_lr = lr

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, _ = self.collect_trainable_params(group, self.state)
                for batch in group_tensors_by_device_and_dtype(params, grads):
                    values = self._step_foreach(group, batch['params'], batch['grads'], wd_lr)
                    if group['bf16_sr'] and batch['params'][0].dtype == torch.bfloat16:
                        for p, value in zip(batch['params'], values):
                            copy_stochastic(p, value.float())
                continue

            self._step_per_param(group, wd_lr)

        return loss
