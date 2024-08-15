import math
from typing import Optional, Set

import torch
from torch import distributed as dist
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS


class AdamMini(BaseOptimizer):  # pragma: no cover
    r"""Use Fewer Learning Rates To Gain More.

    :param model: nn.Module. model instance.
    :param model_sharding: bool. set to True if you are using model parallelism with more than 1 GPU, including FSDP
        and zero_1, 2, 3 in Deepspeed. Set to False if otherwise.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param num_embeds: int. number of embedding dimensions. could be unspecified if you are training non-transformer
        models.
    :param num_heads: int. number of attention heads. could be unspecified if you are training non-transformer models.
    :param num_query_groups: Optional[int]. number of query groups in Group Query Attention (GQA). if not specified, it
        will be equal to num_heads. could be unspecified if you are training non-transformer models.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.0,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.1,
        model_sharding: bool = False,
        num_embeds: int = 2048,
        num_heads: int = 32,
        num_query_groups: Optional[int] = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(num_embeds, 'num_embeds')
        self.validate_non_negative(num_heads, 'num_heads')
        self.validate_non_negative(eps, 'eps')

        self.num_query_groups: int = num_query_groups if num_query_groups is not None else num_embeds
        self.validate_mod(num_embeds, self.num_query_groups)

        self.world_size: int = torch.cuda.device_count()

        self.model = model
        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads

        self.embed_blocks: Set[str] = {'embed', 'embd', 'wte', 'lm_head.weight', 'output.weight'}
        self.qk_blocks: Set[str] = {'k_proj.weight', 'q_proj.weight', 'wq.weight', 'wk.weight'}

        groups = self.get_optimizer_groups(weight_decay)

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'eps': eps}
        super().__init__(groups, defaults)

    def __str__(self) -> str:
        return 'AdamMini'

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

            if 'attn.attn.weight' in name or 'attn.qkv.weight' in name:
                group['n_head'] = self.num_heads
                group['q_per_kv'] = self.num_embeds // self.num_query_groups

            groups.append(group)

        return groups

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['m'] = torch.zeros_like(p, dtype=torch.float32)
                state['v'] = torch.zeros_like(p, dtype=torch.float32)

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
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(head, 1).mul_(m)

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
        num_heads: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state['m'] = torch.zeros_like(p, dtype=torch.float32).view(num_heads, q_per_kv + 2, -1)
            state['v_mean'] = torch.zeros(num_heads, q_per_kv + 2, device=state['m'].device)

        m, v = state['m'], state['v_mean']

        grad = grad.view(num_heads, q_per_kv + 2, -1)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=2).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(num_heads, q_per_kv + 2, -1).mul_(m)

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
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        stepsize = (1 / bias_correction1) / h

        update = m * stepsize

        p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

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

                grad = grad.to(torch.float32)

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
                        group['n_head'],
                        group['q_per_kv'],
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
