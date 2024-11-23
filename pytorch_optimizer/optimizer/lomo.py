import math
import os
from typing import Any, Callable, List, Optional

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import DEFAULTS
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler
from pytorch_optimizer.optimizer.utils import has_overflow, is_deepspeed_zero3_enabled


class LOMO(BaseOptimizer):
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
        **kwargs,
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

        p0: torch.Tensor = next(iter(self.model.parameters()))

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

                if (self.loss_scaler and self.loss_scaler.has_overflow_serial) or has_overflow(p.grad):
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

                if (self.loss_scaler and self.loss_scaler.has_overflow_serial) or has_overflow(p.grad):
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


class AdaLOMO(BaseOptimizer):
    r"""Low-memory Optimization with Adaptive Learning Rate.

    :param model: nn.Module. pytorch model.
    :param lr: float. learning rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param loss_scale: float. loss scale.
    :param clip_threshold: float. threshold of root-mean-square of final gradient update.
    :param decay_rate: float. coefficient used to compute running averages of square gradient.
    :param clip_grad_norm: Optional[float]. clip grad norm.
    :param clip_grad_value: Optional[float]. clip grad value.
    :param eps1: float. term added to the denominator to improve numerical stability.
    :param eps2: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_scale: float = 2.0 ** 10,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        **kwargs,
    ) -> None:  # fmt: skip
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(loss_scale, 'loss_scale')
        self.validate_non_negative(clip_threshold, 'clip_threshold')
        self.validate_non_negative(clip_grad_norm, 'clip_grad_norm')
        self.validate_non_negative(clip_grad_value, 'clip_grad_value')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.eps1 = eps1
        self.eps2 = eps2

        self.num_steps: int = 0
        self.gather_norm: bool = False
        self.grad_norms: List[torch.Tensor] = []
        self.clip_coef: Optional[float] = None

        self.local_rank: int = int(os.environ.get('LOCAL_RANK', 0))
        self.zero3_enabled: bool = is_deepspeed_zero3_enabled()

        self.grad_func: Callable[[Any], Any] = self.fuse_update_zero3() if self.zero3_enabled else self.fuse_update()

        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}

        self.initialize_states()

        defaults: DEFAULTS = {
            'lr': lr,
            'weight_decay': weight_decay,
            'clip_grad_norm': clip_grad_norm,
            'clip_grad_value': clip_grad_value,
            'eps1': eps1,
            'eps2': eps2,
        }
        super().__init__(self.model.parameters(), defaults)

    def __str__(self) -> str:
        return 'AdaLOMO'

    def initialize_states(self) -> None:
        for n, p in self.model.named_parameters():
            if self.zero3_enabled:  # pragma: no cover
                if len(p.ds_shape) == 1:
                    self.exp_avg_sq[n] = torch.zeros(p.ds_shape[0], dtype=torch.float32, device=p.device)
                else:
                    self.exp_avg_sq_row[n] = torch.zeros(p.ds_shape[0], dtype=torch.float32, device=p.device)
                    self.exp_avg_sq_col[n] = torch.zeros(p.ds_shape[1], dtype=torch.float32, device=p.device)
            elif len(p.shape) == 1:
                self.exp_avg_sq[n] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
            else:
                self.exp_avg_sq_row[n] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                self.exp_avg_sq_col[n] = torch.zeros(p.shape[1], dtype=torch.float32, device=p.device)

            if p.requires_grad:
                p.register_hook(self.grad_func)

    @torch.no_grad()
    def reset(self):
        pass

    def fuse_update(self) -> Callable[[Any], Any]:
        @torch.no_grad()
        def func(x: Any) -> Any:
            for n, p in self.model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                grad_fp32 = p.grad.to(torch.float32)
                p.grad = None

                if self.loss_scale:
                    grad_fp32.div_(self.loss_scale)

                if self.gather_norm:
                    self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                else:
                    if self.clip_grad_value is not None and self.clip_grad_value > 0.0:
                        grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                    if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is not None:
                        grad_fp32.mul_(self.clip_coef)

                    beta2_t: float = 1.0 - math.pow(
                        self.num_steps, self.decay_rate if self.num_steps > 0 else -self.decay_rate
                    )

                    update = grad_fp32.pow(2).add_(self.eps1)

                    if len(p.shape) > 1:
                        self.exp_avg_sq_row[n].mul_(beta2_t).add_(update.mean(dim=-1), alpha=1.0 - beta2_t)
                        self.exp_avg_sq_col[n].mul_(beta2_t).add_(update.mean(dim=-2), alpha=1.0 - beta2_t)

                        self.approximate_sq_grad(self.exp_avg_sq_row[n], self.exp_avg_sq_col[n], update)
                        update.mul_(grad_fp32)
                    else:
                        self.exp_avg_sq[n].mul_(beta2_t).add_(update, alpha=1.0 - beta2_t)
                        update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                    update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                    p_fp32 = p.to(torch.float32)
                    p_rms = torch.norm(p_fp32, 2.0) / math.sqrt(p.numel())

                    lr = self.lr * max(self.eps2, p_rms)

                    self.apply_weight_decay(
                        p,
                        grad_fp32,
                        lr,
                        self.weight_decay,
                        weight_decouple=True,
                        fixed_decay=False,
                    )

                    p_fp32.add_(grad_fp32, alpha=-lr)
                    p.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self) -> Callable[[Any], Any]:  # pragma: no cover
        @torch.no_grad()
        def func(x: torch.Tensor) -> torch.Tensor:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue

                all_reduce(p.grad, op=ReduceOp.AVG, async_op=False)

                grad_fp32 = p.grad.to(torch.float32)
                p.grad = None

                if self.loss_scale:
                    grad_fp32.div_(self.loss_scale)

                if self.gather_norm:
                    self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                else:
                    partition_size: int = p.ds_tensor.numel()
                    start = partition_size * self.local_rank
                    end = min(start + partition_size, grad_fp32.numel())

                if self.clip_grad_value is not None:
                    grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                    grad_fp32.mul_(self.clip_coef)

                beta2_t: float = 1.0 - math.pow(
                    self.num_steps, self.decay_rate if self.num_steps > 0 else -self.decay_rate
                )

                update = grad_fp32.pow(2).add_(self.eps1)

                if len(p.ds_shape) > 1:
                    self.exp_avg_sq_row[n].mul_(beta2_t).add_(update.mean(dim=-1), alpha=1.0 - beta2_t)
                    self.exp_avg_sq_col[n].mul_(beta2_t).add_(update.mean(dim=-2), alpha=1.0 - beta2_t)

                    self.approximate_sq_grad(self.exp_avg_sq_row[n], self.exp_avg_sq_col[n], update)
                    update.mul_(grad_fp32)
                else:
                    self.exp_avg_sq[n].mul_(beta2_t).add_(update, alpha=1.0 - beta2_t)
                    update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                one_dim_update = update.view(-1)
                partitioned_update = one_dim_update.narrow(0, start, end - start)

                param_fp32 = p.ds_tensor.to(torch.float32)
                partitioned_p = param_fp32.narrow(0, 0, end - start)

                p_rms = torch.norm(partitioned_p, 2.0).pow_(2)
                all_reduce(p_rms, op=ReduceOp.SUM)
                p_rms.div_(p.ds_numel).sqrt_()

                lr = self.lr * max(self.eps2, p_rms)

                self.apply_weight_decay(
                    partitioned_p,
                    grad_fp32,
                    lr,
                    self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                )

                partitioned_p.add_(partitioned_update, alpha=-lr)

                p.ds_tensor[: end - start] = partitioned_p

            return x

        return func

    def fused_backward(self, loss, lr: float) -> None:
        self.lr = lr

        if self.loss_scale:
            loss = loss * self.loss_scale

        self.num_steps += 1

        loss.backward()

        self.grad_func(0)

    def grad_norm(self, loss) -> None:
        self.gather_norm = True
        self.grad_norms = []

        if self.loss_scale:
            loss = loss * self.loss_scale

        loss.backward(retain_graph=True)

        self.grad_func(0)

        with torch.no_grad():
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = torch.clamp(float(self.clip_grad_norm) / (total_norm + 1e-6), max=1.0)

        self.gather_norm = False
