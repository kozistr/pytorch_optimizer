import argparse
import gc
import json
import statistics
import time

import torch

from pytorch_optimizer.optimizer.utils.matrix import zero_power_via_newton_schulz_5


def benchmark(args):
    shape = tuple(int(size) for size in args.shape.split(','))
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    g = torch.randn(shape, device='cuda')
    function = zero_power_via_newton_schulz_5
    if args.mode != 'eager':
        options = {'triton.cudagraphs': args.mode == 'cudagraph'}
        if args.emulate_precision_casts:
            options['emulate_precision_casts'] = True
        function = torch.compile(function, fullgraph=True, dynamic=False, options=options)

    with torch.no_grad():
        expected = zero_power_via_newton_schulz_5(g, dtype=dtype)
        torch.cuda.synchronize()
        reserved_before = torch.cuda.memory_reserved()
        started = time.perf_counter()
        output = function(g, dtype=dtype).clone()
        torch.cuda.synchronize()
        first_call_seconds = time.perf_counter() - started
        torch.testing.assert_close(output, expected, rtol=0.05, atol=0.01)
        relative_error = ((output.float() - expected.float()).norm() / expected.float().norm()).item()
        del output
        for _ in range(args.warmup):
            function(g, dtype=dtype)
        torch.cuda.synchronize()

        gpu_times, wall_times = [], []
        for _ in range(args.repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                function(g, dtype=dtype)
            end_event.record()
            end_event.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event) / args.iterations)
            wall_times.append((time.perf_counter() - started) * 1000 / args.iterations)

        gc.collect()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = function(g, dtype=dtype)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated() - allocated_before
        del output
        print(json.dumps({
            'python_torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0),
            'mode': args.mode,
            'shape': shape,
            'dtype': args.dtype,
            'emulate_precision_casts': args.emulate_precision_casts,
            'first_call_seconds': first_call_seconds,
            'cuda_ms': statistics.median(gpu_times),
            'wall_ms': statistics.median(wall_times),
            'peak_call_mib': peak_bytes / 2**20,
            'reserved_before_mib': reserved_before / 2**20,
            'reserved_after_mib': torch.cuda.memory_reserved() / 2**20,
            'relative_l2_error': relative_error,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='256,256')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--mode', choices=['eager', 'compiled', 'cudagraph'], default='eager')
    parser.add_argument('--emulate-precision-casts', action='store_true')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this benchmark.')
    if min(args.iterations, args.repeats) <= 0 or args.warmup < 0:
        parser.error('Iterations and repeats must be positive; warmup must be nonnegative.')
    benchmark(args)


if __name__ == '__main__':
    main()
