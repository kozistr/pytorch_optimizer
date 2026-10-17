import argparse
import gc
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from pytorch_optimizer import create_optimizer, load_optimizer


def benchmark(optimizer_name: str, mode: str, args) -> dict:
    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()

    dtype = getattr(torch, args.dtype)
    shape = tuple(args.shape) if args.shape else (args.elements,)
    params = [torch.nn.Parameter(torch.randn(shape, device='cuda', dtype=dtype)) for _ in range(args.tensors)]
    gradients = [torch.randn_like(parameter) for parameter in params]
    for parameter, gradient in zip(params, gradients):
        parameter.grad = gradient.clone()

    options = {'lr': 1e-3, 'weight_decay': 1e-2}
    signature = inspect.signature(load_optimizer(optimizer_name)).parameters
    if 'num_iterations' in signature:
        options.update(num_iterations=max(1000, args.warmup + args.steps + 1), disable_lr_scheduler=True)
    if 'foreach' in signature:
        options['foreach'] = mode != 'per_param'
    elif mode == 'foreach':
        return {'optimizer': optimizer_name, 'mode': mode, 'status': 'unsupported'}

    compile_kwargs = {'dynamic': False} if args.static else None
    optimizer = create_optimizer(
        torch.nn.ParameterList(params),
        optimizer_name,
        compile=mode == 'compiled',
        compile_kwargs=compile_kwargs,
        **options,
    )

    def step():
        torch._foreach_copy_([parameter.grad for parameter in params], gradients)
        optimizer.step()

    started = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.steps)]
    walls = []
    for start, end in events:
        torch._foreach_copy_([parameter.grad for parameter in params], gradients)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start.record()
        optimizer.step()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - started) * 1000.0)

    torch._foreach_copy_([parameter.grad for parameter in params], gradients)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return {
        'optimizer': optimizer_name,
        'mode': mode,
        'options': options,
        'compile_kwargs': compile_kwargs if mode == 'compiled' else None,
        'status': 'ok',
        'gpu_median_ms': statistics.median(start.elapsed_time(end) for start, end in events),
        'wall_median_ms': statistics.median(walls),
        'step_temporary_mib': (peak - allocated) / 1024**2,
        'state_mib': state_bytes / 1024**2,
        'peak_allocated_mib': peak / 1024**2,
        'warmup_seconds': warmup_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizers', nargs='+', default=['adabelief', 'adan', 'lamb'])
    parser.add_argument(
        '--modes', nargs='+', choices=['per_param', 'foreach', 'compiled'], default=['foreach', 'compiled']
    )
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--tensors', type=int, default=128)
    parser.add_argument('--elements', type=int, default=4096)
    parser.add_argument('--shape', type=int, nargs='+')
    parser.add_argument('--static', action='store_true', help='Compile fixed tensor shapes with dynamic=False.')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('.cache/optimizer-step-benchmark.json'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA-enabled PyTorch.')
    if min(args.tensors, args.elements, args.warmup, args.steps) < 1:
        raise ValueError('Tensor count, element count, warmup, and steps must be positive.')
    if args.shape and min(args.shape) < 1:
        raise ValueError('Shape dimensions must be positive.')

    report = {
        'python': sys.executable,
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'dtype': args.dtype,
        'tensors': args.tensors,
        'elements_per_tensor': math.prod(args.shape) if args.shape else args.elements,
        'shape': args.shape or [args.elements],
        'warmup': args.warmup,
        'steps': args.steps,
        'results': [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}), flush=True)
    for name in args.optimizers:
        for mode in args.modes:
            row = benchmark(name, mode, args)
            report['results'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
