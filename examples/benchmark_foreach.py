import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.utils import logging

from pytorch_optimizer import load_optimizer

MODEL_ID = 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls'
MODEL_REVISION = '6a5829f5079c66e78d911e06fe21931cc00232f7'
OPTIMIZERS = ['adabound', 'adamax', 'adamod', 'diffgrad', 'padam', 'radam', 'yogi']


def prepare_dataset(tokenizer, pairs_file: Path, sequence_length: int, full_length_only: bool):
    pairs = [json.loads(line) for line in pairs_file.read_text(encoding='utf-8').splitlines() if line.strip()]

    prefix = (
        '<|im_start|>system\nDecide whether the document is relevant to the query. Answer yes or no.'
        '<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    dataset = []

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

        dataset.append({'input_ids': query_ids + document_ids + suffix_ids, 'labels': float(pair['label'])})

    return dataset


class StepTimer(TrainerCallback):
    def __init__(self, warmup_steps: int, accumulation_steps: int):
        self.warmup_steps = warmup_steps

        self.micro_events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(accumulation_steps)
        ]
        self.start, self.optimizer_start, self.optimizer_end, self.end = [
            torch.cuda.Event(enable_timing=True) for _ in range(4)
        ]

        self.timings = {'forward_ms': [], 'backward_ms': [], 'optimizer_ms': [], 'total_ms': [], 'wall_ms': []}
        self.losses = []
        self.step_losses = []
        self.micro_step = 0

        self.warmup_seconds = 0.0
        self.started = self.previous_end = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.previous_end = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self.micro_step = 0
        self.step_losses.clear()

        self.start.record()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.optimizer_start.record()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.optimizer_end.record()

    def on_step_end(self, args, state, control, **kwargs):
        self.end.record()
        torch.cuda.synchronize()
        finished = time.perf_counter()

        loss = torch.stack(self.step_losses).mean().item()
        if not math.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {state.global_step}: {loss}')

        if state.global_step == self.warmup_steps:
            self.warmup_seconds = finished - self.started
            torch.cuda.reset_peak_memory_stats()

        if state.global_step > self.warmup_steps:
            events = self.micro_events[: self.micro_step]
            self.timings['forward_ms'].append(sum(start.elapsed_time(end) for start, end, _ in events))
            self.timings['backward_ms'].append(sum(end.elapsed_time(backward) for _, end, backward in events))
            self.timings['optimizer_ms'].append(self.optimizer_start.elapsed_time(self.optimizer_end))
            self.timings['total_ms'].append(self.start.elapsed_time(self.end))
            self.timings['wall_ms'].append((finished - self.previous_end) * 1000.0)
            self.losses.append(loss)

        self.previous_end = time.perf_counter()


class BenchmarkTrainer(Trainer):
    def __init__(self, *args, timer: StepTimer, **kwargs):
        super().__init__(*args, callbacks=[timer], **kwargs)

        self.timer = timer
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        start, end, _ = self.timer.micro_events[self.timer.micro_step]
        start.record()

        labels = inputs.pop('labels')
        outputs = model(**inputs, use_cache=False)
        loss = binary_cross_entropy_with_logits(outputs.logits.flatten().float(), labels.float())

        end.record()
        self.timer.step_losses.append(loss.detach())

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.timer.micro_events[self.timer.micro_step][2].record()
        self.timer.micro_step += 1

        return loss


def benchmark_case(model, tokenizer, dataset, optimizer_cls, foreach: bool, args):
    def collate(features):
        batch = tokenizer.pad(features, padding='max_length', max_length=args.sequence_length, return_tensors='pt')

        if args.full_length_only:
            batch.pop('attention_mask', None)

        return batch

    optimizer = optimizer_cls(model.parameters(), lr=args.lr, foreach=foreach)
    timer = StepTimer(args.warmup_steps, args.accumulation_steps)

    training_args = TrainingArguments(
        output_dir=str(args.output.parent / 'foreach-training'),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        max_steps=args.warmup_steps + args.steps,
        learning_rate=args.lr,
        lr_scheduler_type='constant',
        max_grad_norm=0.0,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_drop_last=True,
        logging_strategy='no',
        logging_nan_inf_filter=False,
        save_strategy='no',
        report_to='none',
        disable_tqdm=True,
        seed=42,
        data_seed=42,
    )

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
        optimizers=(optimizer, None),
        timer=timer,
    )

    trainer.train()
    trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

    if len(timer.losses) != args.steps:
        raise RuntimeError('Trainer did not complete the requested number of measured optimizer updates.')

    tensor_states = [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]

    return {
        'optimizer': optimizer_cls.__name__,
        'mode': 'foreach' if foreach else 'per_param',
        'foreach': foreach,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'sequence_length': args.sequence_length,
        'mean_ms': {name: statistics.mean(values) for name, values in timer.timings.items()},
        'std_ms': {name: statistics.pstdev(values) for name, values in timer.timings.items()},
        'median_ms': {name: statistics.median(values) for name, values in timer.timings.items()},
        'pairs_per_second': (
            args.batch_size * args.accumulation_steps * 1000.0 / statistics.mean(timer.timings['wall_ms'])
        ),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'warmup_seconds': timer.warmup_seconds,
        'first_timed_loss': timer.losses[0],
        'final_loss': timer.losses[-1],
        'state_dtypes': sorted({str(value.dtype) for value in tensor_states}),
        'losses': timer.losses,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, padding_side='left',
        local_files_only=args.local_files_only,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', local_files_only=args.local_files_only,
    )

    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.cuda()

    dataset = prepare_dataset(tokenizer, args.pairs_file, args.sequence_length, args.full_length_only)
    effective_batch_size = args.batch_size * args.accumulation_steps
    pairs_used = len(dataset) // effective_batch_size * effective_batch_size
    if pairs_used == 0:
        raise ValueError('The pairs file must contain at least one complete effective batch.')

    dataset = dataset[:pairs_used]

    report = {
        'model': args.model,
        'revision': args.revision,
        'architecture': model.__class__.__name__,
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'dtype': 'bfloat16 parameters, gradients, and optimizer states; float32 BCE loss',
        'gpu': torch.cuda.get_device_name(),
        'python': sys.executable,
        'torch': torch.__version__,
        'trainer': 'transformers.Trainer',
        'attention': 'flash_attention_2',
        'full_length_only': args.full_length_only,
        'pairs_used': pairs_used,
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

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            row = benchmark_case(model, tokenizer, dataset, load_optimizer(optimizer_name), mode == 'foreach', args)
            report['results'].append(row)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({key: value for key, value in row.items() if key != 'losses'}), flush=True)


if __name__ == '__main__':
    main()
