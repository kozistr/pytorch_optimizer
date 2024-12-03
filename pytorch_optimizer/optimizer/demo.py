import math
from importlib.util import find_spec
from typing import List, Optional

import torch
from torch.distributed import ProcessGroup, all_gather, get_world_size

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, LOSS, PARAMETERS

HAS_EINOPS: bool = find_spec('einops') is not None

if HAS_EINOPS:  # pragma: ignore
    from einops import rearrange


class TransformDCT:
    r"""TransformDCT."""

    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm: str = 'ortho'):
        if not HAS_EINOPS:
            raise ImportError('You need to install `einops` to use `TransformDCT`')

        self.target_chunk = target_chunk

        self.shape_dict = {}
        self.f_dict = {}
        self.b_dict = {}

        for group in param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    sc = get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    if sc not in self.f_dict:
                        i = torch.eye(sc)
                        self.f_dict[sc] = dct(i, norm=norm).to(p.dtype).to(p.device)
                        self.b_dict[sc] = inverse_dct(i, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x: torch.Tensor, b: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        if d is None:
            return torch.einsum('...ij, jb -> ...ib', x, b)
        return torch.einsum('...ijkl, jb, ld -> ...ikbd', x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x: torch.Tensor, b: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        if d is None:
            return torch.einsum('...ij, jb -> ...ib', x, b)
        return torch.einsum('...ijkl, kb, ld -> ...ibjd', x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n1 = self.shape_dict[x.shape[0]]
        n1w = self.f_dict[n1].to(x.device)
        self.f_dict[n1] = n1w

        if len(x.shape) > 1:
            n2 = self.shape_dict[x.shape[1]]
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n2] = n2w

            x = rearrange(x, '(y h) (x w) -> y h x w', h=n1, w=n2)
            return self.einsum_2d(x, n1w, n2w)

        x = rearrange(x, '(x w) -> x w', w=n1)
        return self.einsum_2d(x, n1w)

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.einsum_2d_t(x, n1w, n2w)
            return rearrange(x, 'y h x w -> (y h) (x w)')

        n1 = x.shape[1]
        n1w = self.b_dict[n1].to(x.device)
        self.b_dict[n1] = n1w

        x = self.einsum_2d_t(x, n1w)
        return rearrange(x, 'x w -> (x w)')


class CompressDCT:
    r"""CompressDCT."""

    @torch.no_grad()
    def __init__(self):
        if not HAS_EINOPS:
            raise ImportError('You need to install `einops` to use `CompressDCT`')

    @staticmethod
    def clamp_top_k(x: torch.Tensor, top_k: int) -> int:
        if top_k > x.shape[-1]:
            return x.shape[-1]
        if top_k < 1:
            return 1
        return top_k

    @torch.no_grad()
    def compress(self, x: torch.Tensor, top_k: int):
        x_shape = x.shape
        if len(x.shape) > 2:
            x = rearrange(x, 'y x h w -> y x (h w)')

        idx = torch.topk(x.abs(), k=self.clamp_top_k(x, top_k), dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)

        return idx, val, x_shape

    @torch.no_grad()
    def decompress(self, p, idx, val, shape):
        x = torch.zeros(shape, device=p.device, dtype=p.dtype)

        if len(shape) > 2:
            x = rearrange(x, 'y x h w -> y x (h w)')

        x.scatter_reduce_(dim=-1, index=idx, src=val, reduce='mean', include_self=False).reshape(shape)

        if len(x.shape) > 2:
            x = rearrange(x, 'y x (h w) -> y x h w', h=shape[2])

        return x

    @torch.no_grad()
    def batch_decompress(self, p, idx, val, shape) -> torch.Tensor:
        idx = torch.concatenate(idx, dim=-1).to(device=p.device)
        val = torch.concatenate(val, dim=-1).to(device=p.device)
        return self.decompress(p, idx, val, shape)


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    r"""Discrete Cosine Transform, Type II (a.k.a. the DCT).

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: torch.Tensor. the input signal.
    :param norm: Optional[str]. the normalization, None or 'ortho'.
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape

    n = x_shape[-1]
    x = x.contiguous().view(-1, n)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v = vc[:, :, 0] * w_r - vc[:, :, 1] * w_i

    if norm == 'ortho':
        v[:, 0] /= math.sqrt(n) * 2
        v[:, 1:] /= math.sqrt(n / 2) * 2

    return 2 * v.view(*x_shape)


def inverse_dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    r"""Get the inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III.

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: torch.Tensor. the input signal.
    :param norm: Optional[str]. the normalization, None or 'ortho'.
    :return: the inverse DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    n = x_shape[-1]

    x_v = x.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        x_v[:, 0] *= math.sqrt(n) * 2
        x_v[:, 1:] *= math.sqrt(n / 2) * 2

    k = torch.arange(x_shape[-1], dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_r = x_v
    v_t_i = torch.cat([x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1)

    v_r = v_t_r * w_r - v_t_i * w_i
    v_i = v_t_r * w_i + v_t_i * w_r

    v = torch.cat([v_r.unsqueeze(2), v_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(v), n=v.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :n - (n // 2)]  # fmt: skip
    x[:, 1::2] += v.flip([1])[:, :n // 2]  # fmt: skip

    return x.view(*x_shape)


def get_prime_divisors(n: int) -> List[int]:
    r"""Get prime divisors."""
    divisors = []

    while n % 2 == 0:
        divisors.append(2)
        n //= 2

    while n % 3 == 0:
        divisors.append(3)
        n //= 3

    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6

    if n > 1:
        divisors.append(n)

    return divisors


def get_divisors(n: int) -> List[int]:
    r"""Get divisors."""
    divisors = []

    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = get_prime_divisors(n)

        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime

            for i in range(slice_len):
                divisors.append(divisors[i] * factor)

            last_prime = prime

        divisors.sort()

    return divisors


def get_smaller_split(n: int, close_to: int) -> int:
    r"""Get smaller split."""
    all_divisors = get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


class DeMo(torch.optim.SGD, BaseOptimizer):  # pragma: no cover
    r"""Decoupled Momentum Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param compression_decay: float. compression_decay.
    :param compression_top_k: int. compression_top_k.
    :param compression_chunk: int. compression_chunk.
    :param weight_decay: float. weight decay (L2 penalty).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        compression_decay: float = 0.999,
        compression_top_k: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        process_group: Optional[ProcessGroup] = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(compression_decay, 'compression_decay', 0.0, 1.0, range_type='[)')
        self.validate_positive(compression_top_k, 'compression_top_k')
        self.validate_positive(compression_chunk, 'compression_chunk')

        self.weight_decay = weight_decay

        self.compression_decay = compression_decay
        self.compression_top_k = compression_top_k
        self.compression_chunk = compression_chunk
        self.process_group = process_group

        self.data_transmit: int = 0
        self.data_receive: int = 0

        super().__init__(
            params,
            lr=lr,
            foreach=False,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            weight_decay=0.0,
            **kwargs,
        )

        self.demo_state = {}
        self.init_demo_states()
        self.init_parameters()

        self.default_dtype: torch.dtype = self.find_dtype()
        self.transform = TransformDCT(self.param_groups, self.compression_chunk, norm='ortho')
        self.compress = CompressDCT()

    def __str__(self) -> str:
        return 'DeMo'

    def find_dtype(self) -> torch.dtype:
        r"""Return dtype of the parameter."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    return p.dtype
        return torch.float32

    def init_demo_states(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.demo_state[p] = {}

    def init_parameters(self) -> None:
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.demo_state.get(p, {})

                    state['delta'] = torch.zeros_like(p)

    def demo_all_gather(self, sparse_idx, sparse_val):
        world_size: int = get_world_size() if self.process_group is None else self.process_group.size()

        sparse_idx_list = [torch.zeros_like(sparse_idx) for _ in range(world_size)]
        sparse_val_list = [torch.zeros_like(sparse_val) for _ in range(world_size)]

        sparse_idx_handle = all_gather(sparse_idx_list, sparse_idx, group=self.process_group, async_op=True)
        sparse_val_handle = all_gather(sparse_val_list, sparse_val, group=self.process_group, async_op=True)

        sparse_idx_handle.wait()
        sparse_val_handle.wait()

        return sparse_idx_list, sparse_val_list

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        self.data_transmit = 0
        self.data_receive = 0

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.demo_state.get(p, {})

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=lr,
                    weight_decay=self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                )

                if self.compression_decay != 1:
                    state['delta'].mul_(self.compression_decay)

                state['delta'].add_(grad, alpha=lr)

                sparse_idx, sparse_val, x_shape = self.compress.compress(
                    self.transform.encode(state['delta']), self.compression_top_k
                )

                transmit_grad = self.transform.decode(self.compress.decompress(p, sparse_idx, sparse_val, x_shape))

                state['delta'].sub_(transmit_grad)

                sparse_idx_gather, sparse_val_gather = self.demo_all_gather(sparse_idx, sparse_val)

                self.data_transmit += sparse_idx.nbytes + sparse_val.nbytes
                for si, v in zip(sparse_idx_gather, sparse_val_gather):
                    self.data_receive += si.nbytes + v.nbytes

                new_grad = self.transform.decode(
                    self.compress.batch_decompress(p, sparse_idx_gather, sparse_val_gather, x_shape)
                )

                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)

                p.grad.sign_()

        return super().step(closure)
