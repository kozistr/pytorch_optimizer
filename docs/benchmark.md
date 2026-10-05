# Optimizer benchmark

Compare per-parameter, foreach, compiled, and fused optimizer updates with Transformers `Trainer`.
The benchmark uses
[`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls), which has
595,777,536 trainable parameters.
It trains in BF16 and computes binary cross-entropy (BCE) loss in FP32.

## Run

Install CUDA-enabled PyTorch, Transformers, Accelerate, FlashAttention 2, and a CUDA toolchain for `torch.compile`.
Prepare a JSONL file with query/document pairs and binary labels:

```json
{"query": "What causes rainfall?", "document": "Rain forms when atmospheric water condenses.", "label": 1}
```

```bash
python -m examples.benchmark --pairs-file train-pairs.jsonl \
    --batch-size 16 --full-length-only --gradient-checkpointing --steps 20
```

Select optimizers with `--optimizers radam yogi adamw`.
Select update modes with `--modes per_param foreach compiled fused`.
Read the results in `.cache/optimizer-benchmark.json`.

## Results

The recorded runs use these settings:

- Hardware: RTX 5060 with 8 GB of memory.
- Software: Python 3.12, PyTorch 2.14.1+cu132, Transformers 5.18.0, and FlashAttention 2.8.3.
- Training: 256 tokens, a learning rate of 1e-4, and gradient checkpointing.
- Timing: 10 warmup updates and 20 timed updates.
- SciFact training pairs: 1,680 for batch size 16 and 1,664 for batch size 64.

Each case starts from the same weights and random seed.

Each cell lists the median GPU time in milliseconds and the **speedup relative to per-parameter updates**.
The per-parameter baseline is **1×**.
A dash (`—`) marks an unsupported mode.
Compiled mode compiles foreach updates.
The full training step includes the forward pass, backward pass, and optimizer update.

### Optimizer update

| Optimizer | Batch | Per-param (ms) | Foreach (ms, ×) | Compiled (ms, ×) | Fused (ms, ×) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdaBound | 16 | 60.77 | 86.21 (0.70×) | 22.63 (2.69×) | — |
| AdaMax | 16 | 36.81 | 51.12 (0.72×) | 22.33 (1.65×) | — |
| AdaMod | 16 | 64.97 | 85.64 (0.76×) | 28.89 (2.25×) | — |
| DiffGrad | 16 | 70.19 | 96.49 (0.73×) | 28.51 (2.46×) | — |
| PAdam | 16 | 39.66 | 56.56 (0.70×) | 21.61 (1.84×) | — |
| RAdam | 16 | 43.11 | 51.11 (0.84×) | 22.70 (1.90×) | — |
| Yogi | 16 | 57.49 | 78.60 (0.73×) | 34.19 (1.68×) | — |
| Native AdamW | 16 | 40.82 | 58.45 (0.70×) | 22.20 (1.84×) | 22.08 (1.85×) |
| RAdam | 64 | 44.62 | 56.38 (0.79×) | 24.30 (1.84×) | — |

### Full training step

| Optimizer | Batch | Per-param (ms) | Foreach (ms, ×) | Compiled (ms, ×) | Fused (ms, ×) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdaBound | 16 | 798.88 | 825.39 (0.968×) | 763.62 (1.046×) | — |
| AdaMax | 16 | 779.80 | 794.74 (0.981×) | 764.96 (1.019×) | — |
| AdaMod | 16 | 809.30 | 831.62 (0.973×) | 773.09 (1.047×) | — |
| DiffGrad | 16 | 814.72 | 841.44 (0.968×) | 755.75 (1.078×) | — |
| PAdam | 16 | 769.09 | 785.62 (0.979×) | 738.70 (1.041×) | — |
| RAdam | 16 | 789.18 | 794.40 (0.993×) | 777.67 (1.015×) | — |
| Yogi | 16 | 802.17 | 832.20 (0.964×) | 761.53 (1.053×) | — |
| Native AdamW | 16 | 789.94 | 808.53 (0.977×) | 751.42 (1.051×) | 751.61 (1.051×) |
| RAdam | 64 | 3,380.85 | 3,410.31 (0.991×) | 3,372.96 (1.002×) | — |

### Gradient checkpointing

This comparison uses AdaMod foreach updates, batch size 16, and 256 tokens on the same 8 GB GPU.
Each run includes 2 warmup updates and 3 timed updates.

| Checkpointing | Full step (ms) | Peak allocated (GiB) | Speedup vs off |
| --- | ---: | ---: | ---: |
| Off | 28,541.34 | 12.27 | 1.00× |
| On | 822.56 | 6.72 | 34.70× |

The runs use unlocked GPU clocks, so small timing differences vary between runs.
Compiled updates in low precision can change rounding.
The model revision is `6a5829f5079c66e78d911e06fe21931cc00232f7`.
