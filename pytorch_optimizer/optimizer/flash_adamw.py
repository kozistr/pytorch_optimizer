import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT

GROUP_SIZE: int = 32
VALID_MASTER_WEIGHT_BITS: Tuple[Optional[int], ...] = (None, 24, 32)
BITS_TO_BYTES: Dict[Optional[int], int] = {None: 0, 24: 3, 32: 4}
DTYPE_WIDTHS: Dict[torch.dtype, int] = {
    torch.int8: 1,
    torch.int16: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


def _quantized_key(name: str) -> str:
    return f'{name}::quantized'


def _scales_key(name: str) -> str:
    return f'{name}::scales'


def quantize_state(
    tensor: torch.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the states."""
    numel = tensor.numel()
    target_dtype = torch.int8 if signed else torch.uint8
    if numel == 0:
        scales = torch.empty((0,), dtype=torch.float16, device=tensor.device)
        return torch.empty_like(tensor, dtype=target_dtype), scales

    values = tensor.detach().to(torch.float32).reshape(-1)
    if sqrt:
        values = values.clamp_min(0.0).sqrt()

    pad = (-numel) % group_size
    if pad > 0:
        values = torch.cat((values, values.new_zeros(pad)))

    groups = values.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1).clamp_min(1e-12)
    normalized = groups / scales.unsqueeze(1)
    if softsign:
        normalized = 2.0 * normalized / (1.0 + normalized.abs())

    quant_max = 127.0 if signed else 255.0
    quant_min = -127.0 if signed else 0.0
    quantized = torch.round(normalized * quant_max).clamp_(quant_min, quant_max)
    quantized = quantized.reshape(-1)[:numel].reshape_as(tensor).to(target_dtype)
    return quantized, scales.to(torch.float16)


def dequantize_state(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """De-Quantize the states."""
    numel = quantized.numel()
    if numel == 0:
        return torch.empty_like(quantized, dtype=torch.float32)

    values = quantized.to(torch.float32).reshape(-1)
    pad = (-numel) % group_size
    if pad > 0:
        values = torch.cat((values, values.new_zeros(pad)))

    quant_max = 127.0 if signed else 255.0
    groups = values.reshape(-1, group_size) / quant_max
    if softsign:
        groups = groups / (2.0 - groups.abs()).clamp_min(1e-12)

    restored = groups * scales.to(torch.float32).reshape(-1, 1)
    restored = restored.reshape(-1)[:numel].reshape_as(quantized)
    return restored.pow(2) if sqrt else restored


def _state_spec(name: str) -> Tuple[bool, bool, bool]:
    if name == 'exp_avg_sq':
        return False, True, False
    return True, False, True


def materialize_state(state: Dict[str, Any], name: str) -> torch.Tensor:
    """Materialize the states."""
    if _quantized_key(name) in state:
        return dequantize_state(state[_quantized_key(name)], state[_scales_key(name)], *_state_spec(name))
    return state[name].to(torch.float32)


def store_state(state: Dict[str, Any], name: str, tensor: torch.Tensor, quantize: bool, dtype: torch.dtype) -> None:
    """Store the states."""
    if quantize:
        quantized, scales = quantize_state(tensor, *_state_spec(name))
        state[_quantized_key(name)] = quantized
        state[_scales_key(name)] = scales
        state.pop(name, None)
        return

    state[name] = tensor.to(dtype=dtype)
    state.pop(_quantized_key(name), None)
    state.pop(_scales_key(name), None)


def ulp_scale(narrow: torch.Tensor) -> torch.Tensor:
    """Scale the parameter."""
    next_values = torch.nextafter(narrow.abs(), torch.full_like(narrow, float('inf')))
    return next_values.sub(narrow.abs()).to(torch.float32).mul_(0.5).clamp_min_(torch.finfo(torch.float32).tiny)


def compute_ecc_bits(fp32_param: torch.Tensor, narrow_param: torch.Tensor, master_byte_width: int) -> torch.Tensor:
    """Compute ECC bits."""
    if fp32_param.dtype != torch.float32:
        raise ValueError(f'fp32_param must be float32, got {fp32_param.dtype}')
    if narrow_param.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f'narrow_param must be bf16 or fp16, got {narrow_param.dtype}')

    error_bytes = master_byte_width - narrow_param.element_size()
    if error_bytes == 1:
        error_dtype, signed_max = torch.int8, 127.0
    elif error_bytes == 2:
        error_dtype, signed_max = torch.int16, 32767.0
    else:
        raise ValueError(f'unsupported master byte width: {master_byte_width}')

    normalized_error = (fp32_param - narrow_param.to(torch.float32)) / ulp_scale(narrow_param)
    return torch.round(normalized_error.clamp_(-1.0, 1.0) * signed_max).to(error_dtype)


def reconstruct_fp32_param(param: torch.Tensor, error_bits: torch.Tensor) -> torch.Tensor:
    """Reconstruct fp32 parameters."""
    if param.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f'param must be bf16 or fp16, got {param.dtype}')
    if error_bits.dtype == torch.int8:
        signed_max = 127.0
    elif error_bits.dtype == torch.int16:
        signed_max = 32767.0
    else:
        raise ValueError(f'error_bits must be int8 or int16, got {error_bits.dtype}')

    return param.to(torch.float32).add(error_bits.to(torch.float32).div(signed_max).mul(ulp_scale(param)))


class FlashAdamW(BaseOptimizer):
    """FlashOptim-style AdamW with compressed optimizer states.

    The optimizer mirrors FlashOptim's AdamW semantics while keeping the implementation portable for environments where
    Triton kernels are not available. It supports grouped 8-bit optimizer-state compression, compressed state dicts,
    optional low-precision master-weight error correction, and fully LR-decoupled weight decay.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and squared gradient.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Decoupled weight decay coefficient.
        decouple_lr (bool): Scale weight decay by ``lr / initial_lr`` instead of ``lr``.
        quantize (bool): Store Adam moments as grouped 8-bit values plus fp16 scales.
        compress_state_dict (bool): Save quantized states in checkpoints when ``quantize`` is enabled.
        master_weight_bits (Optional[int]): Effective master-weight precision for bf16/fp16 parameters. Supports
            ``None``, ``24``, and ``32``.
        check_numerics (bool): Raise if low-precision parameter updates are unlikely to alter the master weight.
        fused (bool): Placeholder for FlashOptim's Triton fused path. Currently unsupported in this portable backend.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-2,
        decouple_lr: bool = False,
        quantize: bool = True,
        compress_state_dict: bool = True,
        master_weight_bits: Optional[int] = None,
        check_numerics: bool = False,
        fused: bool = False,
        maximize: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(weight_decay, 'weight_decay')

        if master_weight_bits not in VALID_MASTER_WEIGHT_BITS:
            raise ValueError(f'master_weight_bits must be one of {VALID_MASTER_WEIGHT_BITS}')

        if fused:
            raise NotImplementedError('FlashAdamW fused Triton kernels are not available in this portable backend')

        self.maximize = maximize
        self.compress_state_dict = compress_state_dict
        self.check_numerics = check_numerics
        self.master_byte_width = BITS_TO_BYTES[master_weight_bits]
        self.param_absmax: Dict[int, float] = {}

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'decouple_lr': decouple_lr,
            'quantize': quantize,
            'master_byte_width': self.master_byte_width,
            **kwargs,
        }

        super().__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])

        if master_weight_bits is not None and all(
            p.dtype == torch.float32 for group in self.param_groups for p in group['params']
        ):
            raise ValueError('master_weight_bits has no effect when all parameters are fp32')

    def __str__(self) -> str:
        return 'FlashAdamW'

    def maybe_check_numerics(self, p: torch.Tensor, lr: float, master_byte_width: int) -> None:
        if not self.check_numerics or p.dtype == torch.float32 or lr == 0.0:
            return

        max_abs = self.param_absmax.get(id(p))
        if max_abs is None:
            self.param_absmax[id(p)] = max_abs = float(p.detach().abs().max().item()) if p.numel() > 0 else 0.0

        if max_abs <= 0.0 or not math.isfinite(max_abs):
            return

        bits: int = max(DTYPE_WIDTHS[p.dtype], master_byte_width) * 8
        resolution: float = max_abs * 2.0 ** (-(bits - 1))

        if lr * 0.1 < resolution:
            raise ArithmeticError('learning rate is too small to update low-precision FlashAdamW parameters')

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        group.setdefault('initial_lr', group.get('lr'))

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if 'exp_avg' not in state and _quantized_key('exp_avg') not in state:
                store_state(state, 'exp_avg', torch.zeros_like(p, dtype=torch.float32), group['quantize'], p.dtype)
                store_state(state, 'exp_avg_sq', torch.zeros_like(p, dtype=torch.float32), group['quantize'], p.dtype)

            error_bytes = group['master_byte_width'] - DTYPE_WIDTHS[p.dtype]
            if error_bytes > 0 and 'error_bits' not in state:
                error_dtype = torch.int8 if error_bytes == 1 else torch.int16
                state['error_bits'] = torch.zeros_like(p, dtype=error_dtype)

    @staticmethod
    def get_param_fp32(p: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return reconstruct_fp32_param(p, state['error_bits']) if 'error_bits' in state else p.to(torch.float32)

    @staticmethod
    def set_param_fp32(p: torch.Tensor, state: Dict[str, Any], value: torch.Tensor, master_byte_width: int) -> None:
        p.copy_(value.to(p.dtype))
        if 'error_bits' in state:
            state['error_bits'].copy_(compute_ecc_bits(value, p, master_byte_width))

    def recompute_param_stats(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                self.param_absmax[id(p)] = float(p.detach().abs().max().item()) if p.numel() > 0 else 0.0

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                grad = p.grad.to(torch.float32)

                self.maximize_gradient(grad, maximize=self.maximize)

                self.maybe_check_numerics(p, group['lr'], group['master_byte_width'])

                exp_avg = materialize_state(state, 'exp_avg')
                exp_avg_sq = materialize_state(state, 'exp_avg_sq')

                param_fp32 = self.get_param_fp32(p, state)

                self.apply_weight_decay(
                    param_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                    ratio=1.0 / group['initial_lr'],
                )

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denominator = exp_avg_sq.div(bias_correction2).sqrt_().add_(group['eps'])
                param_fp32.addcdiv_(exp_avg.div(bias_correction1), denominator, value=-group['lr'])

                self.set_param_fp32(p, state, param_fp32, group['master_byte_width'])
                store_state(state, 'exp_avg', exp_avg, group['quantize'], p.dtype)
                store_state(state, 'exp_avg_sq', exp_avg_sq, group['quantize'], p.dtype)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        if self.compress_state_dict:
            return state_dict

        state_dict['state'] = {param_id: dict(param_state) for param_id, param_state in state_dict['state'].items()}
        for param_state in state_dict['state'].values():
            for name in ('exp_avg', 'exp_avg_sq'):
                q_key, s_key = _quantized_key(name), _scales_key(name)
                if q_key not in param_state:
                    continue
                param_state[name] = dequantize_state(
                    param_state.pop(q_key), param_state.pop(s_key), *_state_spec(name)
                )

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue

                for name in ('exp_avg', 'exp_avg_sq'):
                    if group['quantize'] and name in state:
                        store_state(state, name, state.pop(name).to(torch.float32), True, p.dtype)
                    elif group['quantize'] and _quantized_key(name) in state:
                        signed, _, _ = _state_spec(name)
                        quantized_dtype = torch.int8 if signed else torch.uint8
                        state[_quantized_key(name)] = state[_quantized_key(name)].to(quantized_dtype)
                        state[_scales_key(name)] = state[_scales_key(name)].to(torch.float16)
                    elif not group['quantize'] and _quantized_key(name) in state:
                        state[name] = dequantize_state(
                            state.pop(_quantized_key(name)), state.pop(_scales_key(name)), *_state_spec(name)
                        ).to(dtype=p.dtype)

    def get_fp32_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: self.get_param_fp32(param.detach(), self.state.get(param, {})).detach().clone()
            for name, param in model.named_parameters()
        }

    @torch.inference_mode()
    def set_fp32_model_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name not in state_dict:
                continue

            state = self.state[param]

            master_byte_width = next(
                group['master_byte_width']
                for group in self.param_groups
                if any(param is grouped_param for grouped_param in group['params'])
            )
            error_bytes = master_byte_width - DTYPE_WIDTHS[param.dtype]
            if error_bytes > 0 and 'error_bits' not in state:
                error_dtype = torch.int8 if error_bytes == 1 else torch.int16
                state['error_bits'] = torch.zeros_like(param, dtype=error_dtype)

            self.set_param_fp32(param, state, state_dict[name].to(torch.float32), master_byte_width)
