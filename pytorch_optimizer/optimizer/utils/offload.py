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
        self.cpu_grad_buffers: dict[torch.Tensor, torch.Tensor] = {}
        self.stream = torch.cuda.Stream()

        self.queue = {}

        def backward_hook(p_cuda: torch.Tensor) -> None:
            if p_cuda.grad is None:
                return

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.cpu_grad_buffers[p_cuda].copy_(p_cuda.grad, non_blocking=True)

            if p_cuda in self.queue:
                del self.queue[p_cuda]

            self.queue[p_cuda] = self.stream.record_event()

            if offload_gradients:
                p_cuda.grad.record_stream(self.stream)
                p_cuda.grad = None

        cpu_groups = []

        for param_group in param_groups:
            group_params = param_group.get('params', None)
            if group_params is None:
                continue

            cpu_params = []

            for p_cuda in group_params:
                p_cpu = torch.empty_like(p_cuda, device='cpu', pin_memory=True)
                self.cpu_grad_buffers[p_cuda] = torch.empty_like(p_cpu, pin_memory=True)

                p_cpu.copy_(p_cuda.detach(), non_blocking=True)
                self.param_cuda2cpu_map[p_cuda] = p_cpu
                cpu_params.append(p_cpu)

                if p_cuda.requires_grad:
                    p_cuda.register_post_accumulate_grad_hook(backward_hook)

            cpu_groups.append({**param_group, 'params': cpu_params})

        self.optimizer = optimizer_class(cpu_groups, **kwargs)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.queue:
            next(reversed(self.queue.values())).synchronize()

        for p_cuda, p_cpu in self.param_cuda2cpu_map.items():
            p_cpu.grad = self.cpu_grad_buffers[p_cuda] if p_cuda in self.queue else None

        self.optimizer.step()

        with torch.cuda.stream(self.stream):
            for p_cuda in self.queue:
                p_cuda.copy_(self.param_cuda2cpu_map[p_cuda], non_blocking=True)

        self.queue.clear()
        torch.cuda.current_stream().wait_stream(self.stream)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p_cuda in self.param_cuda2cpu_map:
            if set_to_none:
                p_cuda.grad = None
            elif p_cuda.grad is not None:
                p_cuda.grad.zero_()

        self.queue.clear()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, list):
            self.optimizer.load_state_dict(state_dict)
            return

        if len(state_dict) != len(self.param_cuda2cpu_map):
            raise ValueError('offloaded optimizer state does not match the current parameters')

        merged = {'state': {}, 'param_groups': []}
        offset = 0

        for group in self.optimizer.state_dict()['param_groups']:
            saved_groups = []

            for index in group['params']:
                saved = state_dict[offset]
                saved_group = saved['param_groups'][0]
                saved_groups.append({key: value for key, value in saved_group.items() if key != 'params'})
                saved_key = saved_group['params'][0]

                if saved_key in saved['state']:
                    merged['state'][index] = saved['state'][saved_key]

                offset += 1

            if saved_groups:
                if any(saved_group != saved_groups[0] for saved_group in saved_groups[1:]):
                    raise ValueError('legacy offloaded parameter options differ within a parameter group')

                group = {**saved_groups[0], 'params': group['params']}

            merged['param_groups'].append(group)

        self.optimizer.load_state_dict(merged)
