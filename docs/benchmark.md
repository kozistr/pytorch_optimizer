# Optimizer benchmark

Compare per-parameter, foreach, compiled, and fused updates while training
[`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls)
with Transformers `Trainer`. Train all 595,777,536 parameters for binary relevance classification:
BF16 parameters, gradients, and moments; FP32 BCE-with-logits loss.

## Run

Use a Python environment with CUDA PyTorch, Transformers, Accelerate, and FlashAttention 2.
Compiled foreach also needs a working CUDA `torch.compile` toolchain and uses
`create_optimizer(model, name, foreach=True, compile=True)`.
Provide query/document pairs in JSONL with binary labels:

```json
{"query": "What causes rainfall?", "document": "Rain forms when atmospheric water condenses.", "label": 1}
{"query": "What causes rainfall?", "document": "Copper conducts electricity.", "label": 0}
```

```bash
python -m examples.benchmark \
    --pairs-file train-pairs.jsonl \
    --batch-size 16 \
    --gradient-checkpointing \
    --full-length-only \
    --warmup-steps 10 \
    --steps 20
```

Select optimizers with `--optimizers radam yogi adamw` and modes with `--modes per_param foreach compiled fused`.
Defaults come from registered optimizers with compiled foreach support, plus native AdamW as a fused reference.
Unsupported modes are recorded explicitly. Compiled mode uses foreach where available.
Increase the effective batch size with `--accumulation-steps 4`.
Omit `--full-length-only` to include shorter, padded examples. The loader preserves queries and scoring suffixes
while truncating documents. Trainer drops incomplete effective batches.

Each run starts from the same weights and seed. The timer excludes warmup, records CUDA events for forward,
backward, optimizer, and full-step time, and synchronizes after each update. Wall time includes data loading and
Trainer work. Inspect `.cache/optimizer-benchmark.json` for timing distributions, pairs/second, GPU memory, and losses.
Compilation is included in warmup time. Graph counts show whether compilation also occurred during measurement.

## Results

We measured on an RTX 5060 (8 GB) with local Python 3.12, PyTorch 2.14.1+cu132, Transformers 5.18.0,
and FlashAttention 2.8.3. Model revision: `6a5829f5079c66e78d911e06fe21931cc00232f7`.

Setup: batch 16, 256 tokens, learning rate 1e-4, checkpointing with `use_reentrant=False`, 10 warmup updates,
and 20 timed updates. Data: 1,680 full-length SciFact train pairs with judged positives and seeded unjudged negatives.

Values are median GPU milliseconds after warmup. Full-step time includes forward, backward, and optimizer work.
The JSON also records means and standard deviations; a long forward pass inflated DiffGrad's compiled mean.
Yogi was rerun after extending its compiled update.

| Optimizer | Per-param update | Foreach update | Compiled update | Per-param step | Foreach step | Compiled step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdaBound | 60.77 | 86.21 | 22.63 | 798.88 | 825.39 | 763.62 |
| AdaMax | 36.81 | 51.12 | 22.33 | 779.80 | 794.74 | 764.96 |
| AdaMod | 64.97 | 85.64 | 28.89 | 809.30 | 831.62 | 773.09 |
| DiffGrad | 70.19 | 96.49 | 28.51 | 814.72 | 841.44 | 755.75 |
| PAdam | 39.66 | 56.56 | 21.61 | 769.09 | 785.62 | 738.70 |
| RAdam | 43.11 | 51.11 | 22.70 | 789.18 | 794.40 | 777.67 |
| Yogi | 57.49 | 78.60 | 34.19 | 802.17 | 832.20 | 761.53 |
| Native AdamW | 40.82 | 58.45 | 22.20 | 789.94 | 808.53 | 751.42 |

Native fused AdamW measured 22.08 ms per update and 751.61 ms per full step. The seven package optimizers expose
foreach and compiled updates; their fused mode is recorded as unsupported. Compilation reduced their update times
by 2.3–3.8× versus eager foreach. Forward and backward limited the full-step improvement.

At batch 64, RAdam used 1,664 pairs with the same 256-token setup:

| Mode | Update (ms) | Full step (ms) |
| --- | ---: | ---: |
| Per-param | 44.62 | 3,380.85 |
| Foreach | 56.38 | 3,410.31 |
| Compiled | 24.30 | 3,372.96 |

Compiled updates keep scalar bookkeeping eager and cache FP32 scalar tensors. Yogi rounds gradient squares before
its compiled sign and moment updates. Fused floating-point intermediates can change rounding relative to eager
low-precision updates. All measured losses were finite, with no graph compilation during timed updates.
Ranking quality and convergence require separate evaluation. Cases ran sequentially with unlocked GPU clocks;
compare small differences with the JSON variability.
