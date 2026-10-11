import argparse
import gc
import importlib.util
import json
import statistics
import sys
import time
from inspect import signature
from pathlib import Path
from typing import Any

import torch

from pytorch_optimizer import load_optimizer
from pytorch_optimizer.optimizer.utils.foreach import compile_foreach_step

WORKLOADS = {'many_small': (64, (4096,)), 'few_large': (4, (2048, 2048))}


def benchmark_case(name: str, workload: str, dtype: torch.dtype, mode: str, args, reference: bool = False) -> dict:
    optimizer_class: Any = load_optimizer(name)
    if reference:
        path = args.reference_root / f'{optimizer_class.__module__.split(".")[-1]}.py'
        spec = importlib.util.spec_from_file_location(f'benchmark_reference_{name}', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load reference implementation: {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        optimizer_class = getattr(module, optimizer_class.__name__)
    arguments = signature(optimizer_class).parameters
    result = {
        'optimizer': name,
        'workload': workload,
        'dtype': str(dtype),
        'mode': mode,
        'implementation': 'reference' if reference else 'current',
    }
    if mode in ('foreach', 'fused') and mode not in arguments:
        return {**result, 'status': 'unsupported'}
    if mode == 'compiled' and not getattr(optimizer_class, '_supports_compiled_foreach', False):
        return {**result, 'status': 'unsupported'}

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    count, shape = WORKLOADS[workload]
    params = [torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True) for _ in range(count)]
    for param in params:
        param.grad = torch.randn_like(param)

    options = {'eps': None} if name == 'prodigy' else {}
    if name == 'qhm':
        options.update(momentum=0.9, nu=0.7)
    if name == 'sgdp':
        options.update(momentum=0.9, nesterov=True)
    if 'foreach' in arguments and mode != 'auto':
        options['foreach'] = mode in ('foreach', 'compiled')
    if 'fused' in arguments:
        options['fused'] = mode == 'fused'
    optimizer = optimizer_class(params, lr=1e-3, **options)
    result['options'] = {'lr': 1e-3, **options}
    if mode == 'compiled':
        optimizer._step_foreach = compile_foreach_step(optimizer._step_foreach)

    for _ in range(args.warmup):
        optimizer.step()
    torch.cuda.synchronize()

    resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    torch.cuda.synchronize()
    temporary_bytes = torch.cuda.max_memory_allocated() - resident_bytes

    cuda_times, wall_times = [], []
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(args.steps):
            optimizer.step()
        end.record()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0 / args.steps)
        cuda_times.append(start.elapsed_time(end) / args.steps)

    return {
        **result,
        'status': 'ok',
        'cuda_ms': statistics.median(cuda_times),
        'wall_ms': statistics.median(wall_times),
        'cuda_samples_ms': cuda_times,
        'wall_samples_ms': wall_times,
        'resident_mib': resident_bytes / 1024 ** 2,
        'temporary_mib': temporary_bytes / 1024 ** 2,
        'peak_mib': (resident_bytes + temporary_bytes) / 1024 ** 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizers', nargs='+',
        default=[
            'qhm', 'aggmo', 'msvag', 'adamg', 'adams', 'swats', 'dadaptlion', 'prodigy', 'adamp', 'sgdp', 'adashift',
        ],
    )
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument('--dtypes', nargs='+', choices=['float32', 'bfloat16'], default=['float32'])
    parser.add_argument(
        '--modes', nargs='+', choices=['per_param', 'auto', 'foreach', 'compiled', 'fused'], default=['auto']
    )
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--label', default='current')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats) < 1:
        parser.error('Warmup, steps, and repeats must be positive.')
    if not torch.cuda.is_available():
        parser.error('Run this benchmark with a CUDA-enabled Python interpreter.')

    report = {
        'label': args.label,
        'python': sys.executable,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(),
        'warmup': args.warmup,
        'steps': args.steps,
        'repeats': args.repeats,
        'workloads': {name: WORKLOADS[name] for name in args.workloads},
        'reference_root': str(args.reference_root) if args.reference_root else None,
        'results': [],
    }
    for name in args.optimizers:
        for workload in args.workloads:
            for dtype in args.dtypes:
                for mode in args.modes:
                    for reference in ([True, False] if args.reference_root else [False]):
                        result = benchmark_case(name, workload, getattr(torch, dtype), mode, args, reference)
                        report['results'].append(result)
                        print(json.dumps(result), flush=True)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
