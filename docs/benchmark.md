# PyTorch Optimizer Foreach Benchmark

**Environment**
- Device: `cuda`
- GPU: NVIDIA GeForce GTX 1060 6GB
- CUDA Version: 12.8
- PyTorch Version: 2.8.0+cu128

## MLP Model (Batch size: 64)

Parameters: 29,375,488

| Optimizer     | Foreach | Avg Time  | Std      | Peak Mem   | Loss     |
|---------------|---------|-----------|----------|------------|----------|
| AdaFactor     | No      | 35.421 ms | 8.907 ms | 329.58 MB  | 0.000321 |
| AdaFactor     | Yes     | 27.109 ms | 0.247 ms | 577.85 MB  | 0.003894 |
| GrokFastAdamW | No      | 30.360 ms | 0.110 ms | 609.55 MB  | 0.000673 |
| GrokFastAdamW | Yes     | 27.240 ms | 0.159 ms | 801.66 MB  | 0.000565 |
| Amos          | No      | 22.081 ms | 0.239 ms | 273.39 MB  | 0.419085 |
| Amos          | Yes     | 20.077 ms | 0.217 ms | 465.53 MB  | 0.369267 |
| Lion          | No      | 20.296 ms | 0.108 ms | 369.43 MB  | 0.006918 |
| Lion          | Yes     | 17.292 ms | 0.096 ms | 465.48 MB  | 0.006869 |
| Tiger         | No      | 14.876 ms | 0.092 ms | 369.43 MB  | 0.629746 |
| Tiger         | Yes     | 13.438 ms | 0.099 ms | 465.48 MB  | 0.556300 |
| Adan          | No      | 40.456 ms | 0.363 ms | 705.61 MB  | 0.447540 |
| Adan          | Yes     | 40.706 ms | 6.065 ms | 1025.78 MB | 0.438211 |
| ADOPT         | No      | 21.104 ms | 0.085 ms | 497.49 MB  | 0.005070 |
| ADOPT         | Yes     | 22.725 ms | 0.160 ms | 689.60 MB  | 0.001950 |
| AdaBelief     | No      | 28.131 ms | 0.109 ms | 497.49 MB  | 0.000009 |
| AdaBelief     | Yes     | 21.932 ms | 0.284 ms | 689.60 MB  | 0.000006 |
| StableAdamW   | No      | 30.821 ms | 0.787 ms | 497.48 MB  | 0.121176 |
| StableAdamW   | Yes     | 28.764 ms | 0.388 ms | 577.54 MB  | 0.000000 |
| Lamb          | No      | 31.722 ms | 1.081 ms | 497.51 MB  | 0.692506 |
| Lamb          | Yes     | 28.144 ms | 0.428 ms | 577.58 MB  | 0.700553 |
| LARS          | No      | 20.467 ms | 0.231 ms | 353.93 MB  | 0.977978 |
| LARS          | Yes     | 18.902 ms | 0.094 ms | 353.93 MB  | 0.978091 |
| SignSGD       | No      | 14.069 ms | 0.112 ms | 369.43 MB  | 0.679428 |
| SignSGD       | Yes     | 12.682 ms | 0.123 ms | 465.48 MB  | 0.661502 |
| SGDW          | No      | 13.339 ms | 0.148 ms | 353.93 MB  | 0.994222 |
| SGDW          | Yes     | 10.332 ms | 0.119 ms | 353.93 MB  | 0.996546 |

## Foreach vs Regular Summary (CUDA)

| Optimizer     | Speedup      | Time (foreach) | Time (regular) | Memory Diff | Mem Diff % |
|---------------|--------------|----------------|----------------|-------------|------------|
| AdaFactor     | 1.31x        | 27.109 ms      | 35.421 ms      | +248.27 MB  | +75.3%     |
| GrokFastAdamW | 1.11x        | 27.240 ms      | 30.360 ms      | +192.11 MB  | +31.5%     |
| Amos          | 1.10x        | 20.077 ms      | 22.081 ms      | +192.15 MB  | +70.3%     |
| Lion          | 1.17x        | 17.292 ms      | 20.296 ms      | +96.05 MB   | +26.0%     |
| Tiger         | 1.11x        | 13.438 ms      | 14.876 ms      | +96.06 MB   | +26.0%     |
| Adan          | 1.01x slower | 40.706 ms      | 40.456 ms      | +320.17 MB  | +45.4%     |
| ADOPT         | 1.08x slower | 22.725 ms      | 21.104 ms      | +192.11 MB  | +38.6%     |
| AdaBelief     | 1.28x        | 21.932 ms      | 28.131 ms      | +192.11 MB  | +38.6%     |
| StableAdamW   | 1.07x        | 28.764 ms      | 30.821 ms      | +80.06 MB   | +16.1%     |
| Lamb          | 1.13x        | 28.144 ms      | 31.722 ms      | +80.07 MB   | +16.1%     |
| LARS          | 1.08x        | 18.902 ms      | 20.467 ms      | +0.00 MB    | +0.0%      |
| SignSGD       | 1.11x        | 12.682 ms      | 14.069 ms      | +96.06 MB   | +26.0%     |
| SGDW          | 1.29x        | 10.332 ms      | 13.339 ms      | +0.00 MB    | +0.0%      |

Average speedup (foreach vs regular): **1.13x**
