import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging

from pytorch_optimizer import load_optimizer

MODEL_ID = 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls'
MODEL_REVISION = '6a5829f5079c66e78d911e06fe21931cc00232f7'
ATTENTION_IMPLEMENTATION = 'flash_attention_2'
OPTIMIZERS = ['adabound', 'adamax', 'adamod', 'diffgrad', 'padam', 'radam', 'yogi']


def prepare_batches(
    tokenizer, pairs_file: Path, batch_size: int, sequence_length: int, full_length_only: bool = False
):
    pairs = [json.loads(line) for line in pairs_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    prefix = (
        '<|im_start|>system\nDecide whether the document is relevant to the query. Answer yes or no.'
        '<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    encoded = []
    labels = []
    for pair in pairs:
        query_ids = tokenizer.encode(
            f'{prefix}<Instruct>: Find relevant scientific abstracts.\n<Query>: {pair["query"]}\n<Document>: ',
            add_special_tokens=False,
        )
        available = sequence_length - len(query_ids) - len(suffix_ids)
        if available <= 0:
            raise ValueError('Sequence length must leave room for the query, document, and scoring suffix.')
        document_ids = tokenizer.encode(pair['document'], add_special_tokens=False)[:available]
        if full_length_only and len(document_ids) < available:
            continue
        encoded.append({'input_ids': query_ids + document_ids + suffix_ids})
        labels.append(pair['label'])

    batches = []
    for start in range(0, len(encoded) - batch_size + 1, batch_size):
        inputs = tokenizer.pad(
            encoded[start : start + batch_size], padding='max_length', max_length=sequence_length, return_tensors='pt'
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        if full_length_only:
            inputs.pop('attention_mask', None)
        targets = torch.tensor(labels[start : start + batch_size], dtype=torch.float32, device='cuda')
        batches.append((inputs, targets))
    if not batches:
        raise ValueError('The pairs file must contain at least one complete batch.')
    return batches


def benchmark_case(model, optimizer_cls, foreach: bool, batches, args):
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, foreach=foreach)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * args.accumulation_steps + 2)]
    timings = {'forward_ms': [], 'backward_ms': [], 'optimizer_ms': [], 'total_ms': [], 'wall_ms': []}
    losses = []
    correct = []
    warmup_started = time.perf_counter()
    warmup_seconds = 0.0

    for step in range(args.warmup_steps + args.steps):
        started = time.perf_counter()
        events[0].record()
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        predictions = []
        for micro_step in range(args.accumulation_steps):
            inputs, labels = batches[(step * args.accumulation_steps + micro_step) % len(batches)]
            logits = model(**inputs, use_cache=False).logits.flatten().float()
            loss = binary_cross_entropy_with_logits(logits, labels)
            events[2 * micro_step + 1].record()
            (loss / args.accumulation_steps).backward()
            events[2 * micro_step + 2].record()
            step_losses.append(loss.detach())
            predictions.append((logits.detach(), labels))
        optimizer.step()
        events[-1].record()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        step_loss = torch.stack(step_losses).mean().item()
        if not torch.isfinite(torch.tensor(step_loss)):
            raise FloatingPointError(f'Non-finite loss at step {step + 1}: {step_loss}')
        if step + 1 == args.warmup_steps:
            warmup_seconds = time.perf_counter() - warmup_started
            torch.cuda.reset_peak_memory_stats()
        if step < args.warmup_steps:
            continue

        timings['forward_ms'].append(
            sum(events[2 * index].elapsed_time(events[2 * index + 1]) for index in range(args.accumulation_steps))
        )
        timings['backward_ms'].append(
            sum(events[2 * index + 1].elapsed_time(events[2 * index + 2]) for index in range(args.accumulation_steps))
        )
        timings['optimizer_ms'].append(events[-2].elapsed_time(events[-1]))
        timings['total_ms'].append(events[0].elapsed_time(events[-1]))
        timings['wall_ms'].append(elapsed_ms)
        losses.append(step_loss)
        accuracies = [((logits >= 0) == labels.bool()).float().mean() for logits, labels in predictions]
        correct.append(torch.stack(accuracies).mean().item())

    tensor_states = [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
    return {
        'optimizer': optimizer_cls.__name__,
        'foreach': foreach,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'sequence_length': args.sequence_length,
        'mean_ms': {name: statistics.mean(values) for name, values in timings.items()},
        'std_ms': {name: statistics.pstdev(values) for name, values in timings.items()},
        'median_ms': {name: statistics.median(values) for name, values in timings.items()},
        'pairs_per_second': args.batch_size * args.accumulation_steps * 1000.0 / statistics.mean(timings['wall_ms']),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'warmup_seconds': warmup_seconds,
        'first_timed_loss': losses[0],
        'final_loss': losses[-1],
        'mean_training_accuracy': statistics.mean(correct),
        'state_dtypes': sorted({str(value.dtype) for value in tensor_states}),
        'losses': losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/huggingface'))
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--pairs-file', type=Path, required=True)
    parser.add_argument('--optimizers', nargs='+', choices=OPTIMIZERS, default=OPTIMIZERS)
    parser.add_argument('--modes', nargs='+', choices=['per_param', 'foreach'], default=['per_param', 'foreach'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=256)
    parser.add_argument('--full-length-only', action='store_true')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('.cache/foreach-qwen.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA.')
    if min(args.batch_size, args.accumulation_steps, args.steps, args.warmup_steps) < 1:
        raise ValueError('Batch size, accumulation, timed steps, and warmup steps must be positive.')

    logging.disable_progress_bar()
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        padding_side='left',
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        local_files_only=args.local_files_only,
    )
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.cuda().train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    batches = prepare_batches(tokenizer, args.pairs_file, args.batch_size, args.sequence_length, args.full_length_only)
    report = {
        'model': args.model,
        'revision': args.revision,
        'architecture': model.__class__.__name__,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'parameter_tensors': len(list(model.parameters())),
        'dtype': 'bfloat16 parameters, gradients, and optimizer states; float32 BCE loss',
        'gpu': torch.cuda.get_device_name(),
        'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        'python': sys.executable,
        'torch': torch.__version__,
        'attention': ATTENTION_IMPLEMENTATION,
        'full_length_only': args.full_length_only,
        'pairs_used': len(batches) * args.batch_size,
        'gradient_checkpointing': args.gradient_checkpointing,
        'warmup_steps': args.warmup_steps,
        'timed_steps': args.steps,
        'lr': args.lr,
        'seed': 42,
        'pairs_file': str(args.pairs_file),
        'results': [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}), flush=True)
    for optimizer_name in args.optimizers:
        for mode in args.modes:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(initial_state)
            torch.manual_seed(42)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = benchmark_case(model, load_optimizer(optimizer_name), mode == 'foreach', batches, args)
            row['mode'] = mode
            report['results'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({key: value for key, value in row.items() if key != 'losses'}), flush=True)


if __name__ == '__main__':
    main()
