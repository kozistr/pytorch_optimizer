# Foreach benchmark

Compare per-parameter and foreach updates while training
[`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls)
with Transformers `Trainer`. Train all 595,777,536 parameters for binary relevance classification:
BF16 parameters, gradients, and optimizer state; FP32 BCE-with-logits loss.

## Run

Use a Python environment with CUDA PyTorch, Transformers, Accelerate, and FlashAttention 2.
Provide query/document pairs in JSONL with binary labels:

```json
{"query": "What causes rainfall?", "document": "Rain forms when atmospheric water condenses.", "label": 1}
{"query": "What causes rainfall?", "document": "Copper conducts electricity.", "label": 0}
```

```bash
python -m examples.benchmark_foreach \
    --pairs-file train-pairs.jsonl \
    --batch-size 16 \
    --gradient-checkpointing \
    --full-length-only \
    --warmup-steps 10 \
    --steps 20
```

Select optimizers with `--optimizers radam yogi`; increase the effective batch size with `--accumulation-steps 4`.
Omit `--full-length-only` to include shorter, padded examples. The loader preserves queries and scoring suffixes
while truncating documents. Trainer drops incomplete effective batches.

Each run starts from the same weights and seed. The timer excludes warmup, records CUDA events for forward,
backward, optimizer, and full-step time, and synchronizes after each update. Wall time includes data loading and
Trainer work. Inspect `.cache/foreach-qwen.json` for timing distributions, pairs/second, GPU memory, and losses.

## Results

We measured on an RTX 5060 (8 GB) with local Python 3.12, PyTorch 2.14.1+cu132, Transformers 5.18.0,
and FlashAttention 2.8.3. Model revision: `6a5829f5079c66e78d911e06fe21931cc00232f7`.

Setup: batch 16, 256 tokens, learning rate 1e-4, checkpointing with `use_reentrant=False`, 10 warmup updates,
and 20 timed updates. Data: 1,680 full-length SciFact train pairs with judged positives and seeded unjudged negatives.

Values are mean GPU milliseconds. Full-step time includes forward, backward, and optimizer work;
step change compares foreach time with per-parameter time.

| Optimizer | Per-param update | Foreach update | Per-param step | Foreach step | Step change |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdaBound | 65.75 | 85.15 | 791.30 | 813.49 | +2.8% |
| AdaMax | 36.43 | 50.27 | 766.65 | 781.43 | +1.9% |
| AdaMod | 69.63 | 84.41 | 803.54 | 817.20 | +1.7% |
| DiffGrad | 75.08 | 94.87 | 807.85 | 828.56 | +2.6% |
| PAdam | 40.54 | 56.84 | 774.95 | 791.04 | +2.1% |
| RAdam | 42.09 | 50.16 | 776.37 | 784.12 | +1.0% |
| Yogi | 57.01 | 75.55 | 790.85 | 809.35 | +2.3% |

Foreach took 1.0–2.8% longer per full training step. We observed finite losses; ranking quality and convergence
require separate evaluation. Forward and backward dominated runtime. Compare timing differences with the JSON
variability: we ran cases in sequence without locking GPU clocks.
