"""Benchmark foreach vs non-foreach optimizer variants.

This script measures the performance difference between foreach (multi-tensor)
operations and standard per-parameter operations in optimizers.

Foreach operations batch tensor computations together, which can provide
significant speedups on CUDA GPUs by:
- Reducing Python loop overhead
- Enabling better kernel fusion
- Improving GPU utilization through batched operations

Usage:
    python examples/benchmark_foreach.py
    python examples/benchmark_foreach.py --device cuda --num-steps 200
    python examples/benchmark_foreach.py --model-type conv --batch-size 32

Note:
    Real speedups (1.1x-1.5x) are primarily observed on CUDA GPUs.
    On CPU, foreach operations fall back to regular loops with minimal difference.
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from pytorch_optimizer import (
    ADOPT,
    LARS,
    SGDW,
    AdaBelief,
    Adan,
    Amos,
    Lamb,
    Lion,
    SignSGD,
    StableAdamW,
    Tiger,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    optimizer_name: str
    foreach: bool
    avg_step_time_ms: float
    std_step_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    allocated_memory_mb: float
    final_loss: float


@dataclass
class ComparisonResult:
    """Comparison between foreach and non-foreach variants."""

    optimizer_name: str
    speedup: float
    time_foreach_ms: float
    time_no_foreach_ms: float
    memory_foreach_mb: float
    memory_no_foreach_mb: float
    memory_diff_mb: float
    memory_diff_pct: float


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    """Get current memory statistics for the device."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return {
            'allocated_mb': torch.cuda.memory_allocated(device) / 1024 / 1024,
            'peak_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved(device) / 1024 / 1024,
        }
    return {'allocated_mb': 0.0, 'peak_mb': 0.0, 'reserved_mb': 0.0}


def reset_memory_stats(device: torch.device) -> None:
    """Reset memory statistics and clear caches."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking with configurable size."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        num_layers: int = 8,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        output_dim = output_dim or input_dim

        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self._num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def num_params(self) -> int:
        return self._num_params


class ConvNet(nn.Module):
    """VGG-style ConvNet for benchmarking."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @property
    def num_params(self) -> int:
        return self._num_params


class TransformerBlock(nn.Module):
    """Simple Transformer block for benchmarking."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.linear2(self.activation(self.linear1(x)))
        return self.norm2(x + ff_out)


class SimpleTransformer(nn.Module):
    """Simple Transformer model for benchmarking."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dim_feedforward) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, d_model)

        self._num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))

    @property
    def num_params(self) -> int:
        return self._num_params


def create_model_and_data(
    model_type: str, batch_size: int, device: torch.device
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor, Callable]:
    """Create model, data, and loss function for benchmarking."""
    if model_type == 'mlp':
        model = SimpleMLP(input_dim=1024, hidden_dim=2048, num_layers=8).to(device)
        data = torch.randn(batch_size, 1024, device=device)
        target = torch.randn(batch_size, 1024, device=device)
        criterion = nn.MSELoss()
    elif model_type == 'mlp_large':
        model = SimpleMLP(input_dim=2048, hidden_dim=4096, num_layers=12).to(device)
        data = torch.randn(batch_size, 2048, device=device)
        target = torch.randn(batch_size, 2048, device=device)
        criterion = nn.MSELoss()
    elif model_type == 'conv':
        model = ConvNet(num_classes=10).to(device)
        data = torch.randn(batch_size, 3, 32, 32, device=device)
        target = torch.randint(0, 10, (batch_size,), device=device)
        criterion = nn.CrossEntropyLoss()
    elif model_type == 'transformer':
        model = SimpleTransformer(d_model=512, num_layers=6).to(device)
        data = torch.randn(batch_size, 128, 512, device=device)
        target = torch.randn(batch_size, 512, device=device)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model, data, target, criterion


def benchmark_optimizer(
    optimizer_cls: Type,
    model_factory: Callable,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: Callable,
    foreach: bool,
    device: torch.device,
    num_steps: int = 100,
    warmup_steps: int = 10,
    **opt_kwargs,
) -> Optional[BenchmarkResult]:
    """Benchmark a single optimizer configuration."""
    reset_memory_stats(device)

    model = model_factory()

    try:
        optimizer = optimizer_cls(model.parameters(), foreach=foreach, **opt_kwargs)
    except TypeError as e:
        if 'foreach' in str(e):
            return None
        raise

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    reset_memory_stats(device)

    step_times = []
    final_loss = 0.0

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(num_steps):
            start_event.record()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            end_event.record()
            torch.cuda.synchronize()
            step_times.append(start_event.elapsed_time(end_event))
            final_loss = loss.item()
    else:
        for _ in range(num_steps):
            start_step = time.perf_counter()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            step_times.append((time.perf_counter() - start_step) * 1000)
            final_loss = loss.item()

    mem_stats = get_memory_stats(device)
    total_time = sum(step_times)
    avg_time = total_time / len(step_times)
    std_time = (sum((t - avg_time) ** 2 for t in step_times) / len(step_times)) ** 0.5

    del model, optimizer
    reset_memory_stats(device)

    return BenchmarkResult(
        optimizer_name=optimizer_cls.__name__,
        foreach=foreach,
        avg_step_time_ms=avg_time,
        std_step_time_ms=std_time,
        total_time_ms=total_time,
        peak_memory_mb=mem_stats['peak_mb'],
        allocated_memory_mb=mem_stats['allocated_mb'],
        final_loss=final_loss,
    )


def run_benchmarks(
    device: torch.device,
    model_type: str = 'mlp',
    num_steps: int = 100,
    warmup_steps: int = 10,
    batch_size: int = 64,
    verbose: bool = True,
) -> Tuple[List[BenchmarkResult], List[ComparisonResult]]:
    """Run benchmarks for all optimizers."""
    results = []
    comparisons = []

    optimizers_config = [
        (Amos, {'lr': 1e-3}),
        (Lion, {'lr': 1e-4}),
        (Tiger, {'lr': 1e-3}),
        (Adan, {'lr': 1e-3}),
        (ADOPT, {'lr': 1e-3}),
        (AdaBelief, {'lr': 1e-3}),
        (StableAdamW, {'lr': 1e-3}),
        (Lamb, {'lr': 1e-3}),
        (LARS, {'lr': 1e-2}),
        (SignSGD, {'lr': 1e-2, 'momentum': 0.9}),
        (SGDW, {'lr': 1e-2, 'momentum': 0.9}),
    ]

    base_model, data, target, criterion = create_model_and_data(model_type, batch_size, device)
    num_params = base_model.num_params
    del base_model
    reset_memory_stats(device)

    def model_factory():
        model, *_ = create_model_and_data(model_type, batch_size, device)
        return model

    if verbose:
        print(f'\nModel: {model_type}, Parameters: {num_params:,}, Batch size: {batch_size}')
        print('-' * 100)
        header = f'{"Optimizer":15} {"Foreach":>8} {"Avg Time":>12} {"Std":>10} {"Peak Mem":>12} {"Loss":>12}'
        print(header)
        print('-' * 100)

    for opt_cls, opt_kwargs in optimizers_config:
        opt_results = {}

        for foreach in [False, True]:
            result = benchmark_optimizer(
                opt_cls,
                model_factory,
                data,
                target,
                criterion,
                foreach=foreach,
                device=device,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                **opt_kwargs,
            )

            if result:
                results.append(result)
                opt_results[foreach] = result

                if verbose:
                    foreach_str = 'Yes' if foreach else 'No'
                    print(
                        f'{result.optimizer_name:15} {foreach_str:>8} '
                        f'{result.avg_step_time_ms:>9.3f} ms '
                        f'{result.std_step_time_ms:>7.3f} ms '
                        f'{result.peak_memory_mb:>9.2f} MB '
                        f'{result.final_loss:>12.6f}'
                    )

        if True in opt_results and False in opt_results:
            r_foreach = opt_results[True]
            r_no_foreach = opt_results[False]

            speedup = r_no_foreach.avg_step_time_ms / r_foreach.avg_step_time_ms
            mem_diff = r_foreach.peak_memory_mb - r_no_foreach.peak_memory_mb
            mem_diff_pct = (mem_diff / r_no_foreach.peak_memory_mb * 100) if r_no_foreach.peak_memory_mb > 0 else 0

            comparisons.append(
                ComparisonResult(
                    optimizer_name=opt_cls.__name__,
                    speedup=speedup,
                    time_foreach_ms=r_foreach.avg_step_time_ms,
                    time_no_foreach_ms=r_no_foreach.avg_step_time_ms,
                    memory_foreach_mb=r_foreach.peak_memory_mb,
                    memory_no_foreach_mb=r_no_foreach.peak_memory_mb,
                    memory_diff_mb=mem_diff,
                    memory_diff_pct=mem_diff_pct,
                )
            )

        if verbose:
            print()

    return results, comparisons


def print_summary(comparisons: List[ComparisonResult], device_name: str) -> None:
    if not comparisons:
        print('No comparison results available.')
        return

    print('\n' + '=' * 100)
    print(f'SUMMARY: Foreach vs Non-Foreach Performance Comparison ({device_name})')
    print('=' * 100)

    header = (
        f'{"Optimizer":15} {"Speedup":>10} {"Time (foreach)":>15} '
        f'{"Time (regular)":>15} {"Memory Diff":>12} {"Mem Diff %":>10}'
    )
    print(header)
    print('-' * 100)

    avg_speedup = 0.0
    for comp in comparisons:
        speedup_str = f'{comp.speedup:.2f}x' if comp.speedup >= 1 else f'{1/comp.speedup:.2f}x slower'
        mem_diff_str = f'{comp.memory_diff_mb:+.2f} MB'
        mem_pct_str = f'{comp.memory_diff_pct:+.1f}%'

        print(
            f'{comp.optimizer_name:15} {speedup_str:>10} '
            f'{comp.time_foreach_ms:>12.3f} ms '
            f'{comp.time_no_foreach_ms:>12.3f} ms '
            f'{mem_diff_str:>12} {mem_pct_str:>10}'
        )
        avg_speedup += comp.speedup

    avg_speedup /= len(comparisons)
    print('-' * 100)
    print(f'{"Average":15} {avg_speedup:.2f}x')
    print('=' * 100)

    print('\nInterpretation:')
    print('  - Speedup > 1.0x means foreach is faster')
    print('  - Memory diff shows additional memory used by foreach (usually minimal)')
    if device_name == 'cpu':
        print('  - NOTE: On CPU, foreach falls back to regular loops (minimal difference expected)')
        print('  - Real speedups (1.1x-1.5x) are observed on CUDA GPUs')


def main():
    parser = argparse.ArgumentParser(description='Benchmark foreach optimizer variants')
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run benchmarks on',
    )
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'mlp_large', 'conv', 'transformer'])
    parser.add_argument('--num-steps', type=int, default=100, help='Number of benchmark steps')
    parser.add_argument('--warmup-steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--all-models', action='store_true', help='Run benchmarks on all model types')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print('=' * 100)
    print('PyTorch Optimizer Foreach Benchmark')
    print('=' * 100)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
        print(f'CUDA Version: {torch.version.cuda}')
    print(f'PyTorch Version: {torch.__version__}')
    if device.type == 'cpu':
        print(f'CPU Threads: {torch.get_num_threads()}')
    print('=' * 100)

    if device.type == 'cpu':
        print('\nWARNING: Running on CPU. Foreach operations fall back to regular loops.')
        print('         For real speedup measurements, run on CUDA GPU.')
        print()

    model_types = ['mlp', 'conv', 'transformer'] if args.all_models else [args.model_type]
    all_comparisons = {}

    for model_type in model_types:
        print(f'\n{"=" * 100}')
        print(f'Benchmarking: {model_type.upper()} Model')
        print('=' * 100)

        _, comparisons = run_benchmarks(
            device=device,
            model_type=model_type,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            verbose=True,
        )
        all_comparisons[model_type] = comparisons
        print_summary(comparisons, str(device))

    if len(model_types) > 1:
        print('\n' + '=' * 100)
        print('OVERALL SUMMARY ACROSS ALL MODELS')
        print('=' * 100)

        optimizer_speedups: Dict[str, List[float]] = {}
        for _, comparisons in all_comparisons.items():
            for comp in comparisons:
                if comp.optimizer_name not in optimizer_speedups:
                    optimizer_speedups[comp.optimizer_name] = []
                optimizer_speedups[comp.optimizer_name].append(comp.speedup)

        print(f'{"Optimizer":15} {"Avg Speedup":>12} {"Min":>10} {"Max":>10}')
        print('-' * 50)
        for opt_name, speedups in optimizer_speedups.items():
            avg = sum(speedups) / len(speedups)
            print(f'{opt_name:15} {avg:>10.2f}x {min(speedups):>8.2f}x {max(speedups):>8.2f}x')

        overall_avg = sum(sum(s) for s in optimizer_speedups.values()) / sum(
            len(s) for s in optimizer_speedups.values()
        )
        print('-' * 50)
        print(f'{"Overall":15} {overall_avg:>10.2f}x')
        print('=' * 100)


if __name__ == '__main__':
    main()
