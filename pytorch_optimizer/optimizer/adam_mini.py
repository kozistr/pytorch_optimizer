import math

import torch
from torch import distributed as dist
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup


class AdamMini(BaseOptimizer):  # pragma: no cover
    """Adam with shared second moment estimates within parameter blocks.

    Args:
        model: Model instance.
        model_sharding: Set to True if you are using model parallelism with more than 1 GPU, including FSDP and
            zero_1, zero_2, zero_3 in DeepSpeed. Set to False otherwise.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        num_embeds: Number of embedding dimensions. Could be unspecified if training non transformer models.
        num_heads: Number of attention heads. Could be unspecified if training non transformer models.
        num_query_groups: Number of query groups in Group Query Attention (GQA). If not specified, defaults to
            num_heads. Could be unspecified for non transformer models.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        model_sharding: bool = False,
        num_embeds: int = 2048,
        num_heads: int = 32,
        num_query_groups: int | None = None,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(num_embeds, 'num_embeds')
        self.validate_positive(num_heads, 'num_heads')
        self.validate_non_negative(eps, 'eps')

        self.num_query_groups: int = num_query_groups if num_query_groups is not None else num_heads
        self.validate_positive(self.num_query_groups, 'num_query_groups')
        self.validate_mod(num_embeds, num_heads)
        self.validate_mod(num_heads, self.num_query_groups)

        # Visible GPUs are not a process group. all_gather below requires
        # dist to be initialized; otherwise a single process that can see
        # several devices calls it and raises.
        if dist.is_available() and dist.is_initialized():
            self.world_size: int = dist.get_world_size()
        else:
            self.world_size = 1

        self.model = model
        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads

        self.embed_blocks: set[str] = {'embed', 'embd', 'wte', 'lm_head.weight', 'output.weight'}
        self.qk_blocks: set[str] = {'k_proj.weight', 'q_proj.weight', 'wq.weight', 'wk.weight'}

        self.maximize = maximize

        groups = self.get_optimizer_groups(weight_decay)

        defaults: Defaults = {'lr': lr, 'betas': betas, 'eps': eps, **kwargs}

        super().__init__(groups, defaults)

    def __str__(self) -> str:
        return 'AdamMini'

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for p, key in zip(group['params'], saved_group['params']):
                for name, value in state_dict['state'].get(key, {}).items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        self.state[p][name] = value.to(device=p.device, dtype=torch.float32)

    def get_optimizer_groups(self, weight_decay: float):
        groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            group = {
                'name': name,
                'params': param,
                'weight_decay': 0.0 if ('norm' in name or 'ln_f' in name) else weight_decay,
            }

            if any(block in name for block in self.qk_blocks):
                group['parameter_per_head'] = self.num_embeds * self.num_embeds // self.num_heads

            groups.append(group)

        return groups

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @staticmethod
    def step_embed(
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state['m'] = torch.zeros_like(p, dtype=torch.float32)
            state['v'] = torch.zeros_like(p, dtype=torch.float32)

        m, v = state['m'], state['v']

        m.lerp_(grad, weight=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        # PyTorch 2.1 CPU addcdiv does not support mixed precision inputs.
        if p.device.type == 'cpu' and p.dtype != m.dtype:
            p.add_(m / h, alpha=-lr / bias_correction1)
        else:
            p.addcdiv_(m, h, value=-lr / bias_correction1)

    @staticmethod
    def step_attn_proj(
        p,
        grad,
        state,
        parameter_per_head: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state['m'] = torch.zeros_like(p, dtype=torch.float32).view(-1, parameter_per_head)
            state['head'] = state['m'].shape[0]
            state['v_mean'] = torch.zeros(state['head'], device=state['m'].device)

        m, v = state['m'], state['v_mean']

        head: int = state['head']
        grad = grad.view(head, parameter_per_head)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=1).to(m.device)
        v.lerp_(tmp_lr.to(dtype=v.dtype), weight=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(head, 1).mul(m)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)

    @staticmethod
    def step_attn(
        p,
        grad,
        state,
        num_query_groups: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state['m'] = torch.zeros_like(p, dtype=torch.float32).view(num_query_groups, q_per_kv + 2, -1)
            state['v_mean'] = torch.zeros(num_query_groups, q_per_kv + 2, device=state['m'].device)

        m, v = state['m'], state['v_mean']

        grad = grad.view(num_query_groups, q_per_kv + 2, -1)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=2).to(m.device)
        v.lerp_(tmp_lr.to(dtype=v.dtype), weight=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = m / (h * bias_correction1).unsqueeze(-1)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)

    def step_lefts(
        self,
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            dim = torch.tensor(p.numel(), device=p.device, dtype=torch.float32)

            reduced: bool = False
            if self.model_sharding and self.world_size > 1:
                tensor_list = [torch.zeros_like(dim) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, dim)

                s, dim = 0, 0
                for d in tensor_list:
                    if d > 0:
                        s += 1
                    dim += d

                if s >= 2:
                    reduced = True

            state['m'] = torch.zeros_like(p, dtype=torch.float32)
            state['v_mean'] = torch.tensor(0.0, device=state['m'].device)
            state['dimension'] = dim
            state['reduced'] = reduced

        tmp_lr = torch.sum(grad * grad)

        if state['reduced']:
            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

        tmp_lr.div_(state['dimension'])

        m, v = state['m'], state['v_mean']

        m.lerp_(grad, weight=1.0 - beta1)
        v.lerp_(tmp_lr.to(dtype=v.dtype), weight=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        stepsize = (1 / bias_correction1) / h

        update = m * stepsize

        p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            name = group['name']

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction2_sq: float = math.sqrt(bias_correction2)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                grad = grad.to(torch.float32)

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                if any(block in name for block in self.embed_blocks):
                    self.step_embed(
                        p, grad, state, group['lr'], beta1, beta2, bias_correction1, bias_correction2_sq, group['eps']
                    )
                elif any(block in name for block in self.qk_blocks):
                    self.step_attn_proj(
                        p,
                        grad,
                        state,
                        group['parameter_per_head'],
                        group['lr'],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group['eps'],
                    )
                elif 'attn.attn.weight' in name or 'attn.qkv.weight' in name:
                    self.step_attn(
                        p,
                        grad,
                        state,
                        self.num_query_groups,
                        self.num_heads // self.num_query_groups,
                        group['lr'],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group['eps'],
                    )
                else:
                    self.step_lefts(
                        p,
                        grad,
                        state,
                        group['lr'],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group['eps'],
                    )

        return loss
