import os
from typing import Any, Callable, List, Optional

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import DEFAULTS
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler
from pytorch_optimizer.optimizer.utils import has_overflow


class LOMO(BaseOptimizer, Optimizer):
    r"""Full Parameter Fine-tuning for Large Language Models with Limited Resources.

    Reference : https://github.com/OpenLMLab/LOMO/blob/main/src/lomo.py
    Check the usage from here : https://github.com/OpenLMLab/LOMO/blob/main/src/lomo_trainer.py

    :param model: nn.Module. pytorch model.
    :param lr: float. learning rate.
    :param clip_grad_norm: Optional[float]. clip grad norm.
    :param clip_grad_value: Optional[float]. clip grad value.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(clip_grad_norm, 'clip_grad_norm')
        self.validate_non_negative(clip_grad_value, 'clip_grad_value')

        self.model = model
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self.local_rank: int = int(os.environ.get('LOCAL_RANK', 0))

        self.gather_norm: bool = False
        self.grad_norms: List[torch.Tensor] = []
        self.clip_coef: Optional[float] = None

        p0: torch.Tensor = list(self.model.parameters())[0]

        self.grad_func: Callable[[Any], Any] = (
            self.fuse_update_zero3() if hasattr(p0, 'ds_tensor') else self.fuse_update()
        )

        self.loss_scaler: Optional[DynamicLossScaler] = None
        if p0.dtype == torch.float16:
            if clip_grad_norm is None:
                raise ValueError(
                    '[-] Loss scaling is recommended to be used with grad norm to get better performance.'
                )

            self.loss_scaler = DynamicLossScaler(init_scale=2 ** 16)  # fmt: skip

        for _, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)

        defaults: DEFAULTS = {'lr': lr}
        super().__init__(self.model.parameters(), defaults)

    def __str__(self) -> str:
        return 'LOMO'

    @torch.no_grad()
    def reset(self):
        pass

    def fuse_update(self) -> Callable[[Any], Any]:
        @torch.no_grad()
        def func(x: Any) -> Any:
            for _, p in self.model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                if self.loss_scaler and self.loss_scaler.has_overflow_serial or has_overflow(p.grad):
                    p.grad = None
                    self.loss_scaler.has_overflow_serial = True
                    break

                grad_fp32 = p.grad.to(torch.float32)
                p.grad = None

                if self.loss_scaler:
                    grad_fp32.div_(self.loss_scaler.loss_scale)

                if self.gather_norm:
                    self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                else:
                    if self.clip_grad_value is not None and self.clip_grad_value > 0.0:
                        grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                    if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is not None:
                        grad_fp32.mul_(self.clip_coef)

                    p_fp32 = p.to(torch.float32)
                    p_fp32.add_(grad_fp32, alpha=-self.lr)
                    p.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self) -> Callable[[Any], Any]:  # pragma: no cover
        @torch.no_grad()
        def func(x: torch.Tensor) -> torch.Tensor:
            for _, p in self.model.named_parameters():
                if p.grad is None:
                    continue

                all_reduce(p.grad, op=ReduceOp.AVG, async_op=False)

                if self.loss_scaler and self.loss_scaler.has_overflow_serial or has_overflow(p.grad):
                    p.grad = None
                    self.loss_scaler.has_overflow_serial = True
                    break

                grad_fp32 = p.grad.to(torch.float32)
                p.grad = None

                param_fp32 = p.ds_tensor.to(torch.float32)
                if self.loss_scaler:
                    grad_fp32.div_(self.loss_scaler.loss_scale)

                if self.gather_norm:
                    self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                else:
                    one_dim_grad_fp32 = grad_fp32.view(-1)

                    partition_size: int = p.ds_tensor.numel()
                    start: int = partition_size * self.local_rank
                    end: int = min(start + partition_size, grad_fp32.numel())

                    partitioned_grad_fp32 = one_dim_grad_fp32.narrow(0, start, end - start)

                    if self.clip_grad_value is not None:
                        partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)

                    if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                        partitioned_grad_fp32.mul_(self.clip_coef)

                    partitioned_p = param_fp32.narrow(0, 0, end - start)
                    partitioned_p.add_(partitioned_grad_fp32, alpha=-self.lr)

                    p.ds_tensor[: end - start] = partitioned_p  # fmt: skip

            return x

        return func

    def fused_backward(self, loss, lr: float):
        self.lr = lr

        if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is None:
            raise ValueError(
                'clip_grad_norm is not None, but clip_coef is None. '
                'Please call optimizer.grad_norm() before optimizer.fused_backward().'
            )

        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale

        loss.backward()

        self.grad_func(0)

    def grad_norm(self, loss):
        self.gather_norm = True
        self.grad_norms = []

        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale

        loss.backward(retain_graph=True)

        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)

            with torch.no_grad():
                for _, p in self.model.named_parameters():
                    p.grad = None
            return

        with torch.no_grad():
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = torch.clamp(float(self.clip_grad_norm) / (total_norm + 1e-6), max=1.0)

        self.gather_norm = False
