# PyTorch Optimizer Foreach Benchmark

`examples/benchmark_foreach.py` compares per-parameter and foreach updates during full training of
[`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls)
with Transformers `Trainer`. It trains all 595,777,536 parameters using binary relevance labels.
Parameters, gradients, and optimizer state use BF16; BCE-with-logits loss uses FP32. There are no FP32 master weights.
The benchmark uses FlashAttention 2 and requires a CUDA PyTorch installation, Transformers, Accelerate, and a
compatible FlashAttention 2 extension in the Python environment running it.

## Run the benchmark

Provide JSONL records with `query`, `document`, and a binary `label`:

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
    --steps 20 \
    --output .cache/foreach-qwen.json
```

By default, this runs AdaBound, AdaMax, AdaMod, DiffGrad, PAdam, RAdam, and Yogi with both update paths.
Use `--optimizers radam yogi` to select cases, or `--accumulation-steps 4` to increase the effective batch size.
The default sequence length is 256 tokens. `--full-length-only` selects documents that fill the available token
budget so training uses dense attention. Omit it to retain shorter documents and exercise padded attention.
The query and classification suffix are preserved when truncating documents. Incomplete effective batches are dropped.

Each case restores the same initial checkpoint and seed. CUDA events measure forward, backward, optimizer,
and total GPU time. Wall time includes data loading and Trainer work between updates; the timer synchronizes
after each update. Warmup updates are excluded. JSON output also contains timing variability, pair throughput,
peak allocated/reserved GPU memory, optimizer-state dtypes, and measured losses.

## Qwen3 training results

The measurements below use an RTX 5060 8 GB, local Python 3.12, PyTorch 2.14.1+cu132, Transformers 5.18.0,
and FlashAttention 2.8.3. Model revision: `6a5829f5079c66e78d911e06fe21931cc00232f7`.
The input contains SciFact train queries with judged positives and seeded unjudged negatives. After selecting
full-length examples and dropping the incomplete batch, 1,680 pairs remain. Batch size is 16, sequence length is
256, learning rate is 1e-4, and gradient checkpointing uses `use_reentrant=False`.
Each case uses 10 warmup updates followed by 20 measured updates.

| Optimizer | Per-param update (ms) | Foreach update (ms) | Per-param step (ms) | Foreach step (ms) | Step change |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdaBound | 65.75 | 85.15 | 791.30 | 813.49 | +2.8% |
| AdaMax | 36.43 | 50.27 | 766.65 | 781.43 | +1.9% |
| AdaMod | 69.63 | 84.41 | 803.54 | 817.20 | +1.7% |
| DiffGrad | 75.08 | 94.87 | 807.85 | 828.56 | +2.6% |
| PAdam | 40.54 | 56.84 | 774.95 | 791.04 | +2.1% |
| RAdam | 42.09 | 50.16 | 776.37 | 784.12 | +1.0% |
| Yogi | 57.01 | 75.55 | 790.85 | 809.35 | +2.3% |

Foreach takes 1.0–2.8% longer per full training step on this workload. All measured losses are finite.
An additional Trainer smoke run with RAdam foreach at batch 64 completed with finite losses and 5,746 MiB
peak allocated GPU memory.

These measurements assess training throughput and finite losses. They do not evaluate ranking quality or
convergence. Forward and backward dominate the full training step, and foreach performance depends on tensor
shapes and hardware. Cases run sequentially on a desktop GPU without locked clocks, so small timing differences
should be interpreted alongside the variability recorded in the JSON output.
