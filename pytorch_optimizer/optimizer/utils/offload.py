import functools
import operator
from typing import cast

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.compatibility import TORCH_VERSION_AT_LEAST_2_4
from pytorch_optimizer.base.type import Closure, Loss, ParamGroup, ParamsT


class CPUOffloadOptimizer:  # pragma: no cover
    """Offload optimizer states and updates to the CPU for single GPU training.

    Transfers gradients to pinned CPU memory and copies updated parameters back to the GPU.

    Reference: https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/cpu_offload.py

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        optimizer_class: Base optimizer class. Defaults to `torch.optim.AdamW`.
        offload_gradients: Free GPU gradients after transfer. Incompatible with gradient accumulation.
        **kwargs (dict): Options for the base optimizer, such as `lr` and `weight_decay`.

    """

    def __init__(
        self,
        params: ParamsT,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        *,
        offload_gradients: bool = False,
        **kwargs,
    ) -> None:
        if optimizer_class is torch.optim.AdamW and TORCH_VERSION_AT_LEAST_2_4 and 'fused' not in kwargs:
            kwargs.update(fused=True)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('optimizer got an empty parameter list')

        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        param_groups = cast(list[ParamGroup], param_groups)

        self.param_cuda2cpu_map: dict[torch.Tensor, torch.Tensor] = {}
        self.optim_dict: dict[torch.Tensor, Optimizer] = {}
        self.stream = torch.cuda.Stream()

        self.queue = {}

        def backward_hook(p_cuda: torch.Tensor) -> None:
            if p_cuda.grad is None:
                return

            p_cpu = self.param_cuda2cpu_map[p_cuda]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                p_cpu.grad.copy_(p_cuda.grad, non_blocking=True)

            if p_cuda in self.queue:
                del self.queue[p_cuda]

            self.queue[p_cuda] = self.stream.record_event()

            if offload_gradients:
                p_cuda.grad.record_stream(self.stream)
                p_cuda.grad = None

        for param_group in param_groups:
            group_params = param_group.get('params', None)
            if group_params is None:
                continue

            for p_cuda in group_params:
                p_cpu = torch.empty_like(p_cuda, device='cpu', pin_memory=True)
                p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)

                p_cpu.copy_(p_cuda.detach(), non_blocking=True)
                self.param_cuda2cpu_map[p_cuda] = p_cpu

                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([{**param_group, 'params': [p_cpu]}], **kwargs)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = None
        if closure is not None:
            loss = closure()

        for p_cuda, grad_d2h_event in self.queue.items():
            grad_d2h_event.synchronize()
            self.optim_dict[p_cuda].step()

            p_cpu = self.param_cuda2cpu_map[p_cuda]
            with torch.cuda.stream(self.stream):
                p_cuda.copy_(p_cpu, non_blocking=True)

        self.queue.clear()

        return loss

    def zero_grad(self, _: bool = True) -> None:
        for p_cuda in self.param_cuda2cpu_map:
            p_cuda.grad = None

    @property
    def param_groups(self):
        return functools.reduce(operator.add, (optim.param_groups for optim in self.optim_dict.values()), [])

    def state_dict(self):
        return [optim.state_dict() for optim in self.optim_dict.values()]

    def load_state_dict(self, state_dict):
        for optim, optim_state_dict in zip(self.optim_dict.values(), state_dict):
            optim.load_state_dict(optim_state_dict)
