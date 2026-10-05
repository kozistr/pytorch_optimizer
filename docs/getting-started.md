# Getting Started

## Installation

Use Python 3.10 or later with PyTorch 2.1 or later. The package installer selects dependencies that support your Python version.

```bash
pip install pytorch-optimizer
```

Install optional integrations from their projects as needed:
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
[Q-GaLore](https://github.com/VITA-Group/Q-GaLore), or
[TorchAO](https://github.com/pytorch/ao).

## Run a training step

Pass the model parameters to an optimizer class. Use the optimizer in a PyTorch training loop:

```python
import torch
from torch import nn
from pytorch_optimizer import AdamP

model = nn.Linear(10, 1)
optimizer = AdamP(model.parameters(), lr=1e-3)
inputs = torch.randn(8, 10)
targets = torch.randn(8, 1)

optimizer.zero_grad()
loss = nn.functional.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
```

## Load an optimizer by name

Use `load_optimizer()` to select an optimizer by its name in a training configuration:

```python
from pytorch_optimizer import load_optimizer

optimizer = load_optimizer('adamp')(model.parameters(), lr=1e-3)
```

Use Torch Hub to load an optimizer class:

```python
optimizer_class = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
optimizer = optimizer_class(model.parameters(), lr=1e-3)
```

## Combine optimizer options

Pass the model to `create_optimizer()` to configure weight decay and optimizer wrappers:

```python
from pytorch_optimizer import create_optimizer

optimizer = create_optimizer(
    model,
    'adamp',
    lr=1e-3,
    weight_decay=1e-3,
    use_gc=True,
    use_lookahead=True,
)
```

Check the [optimizer reference](optimizer.md) for options and arguments specific to each optimizer.

## Compile optimizer steps

Set `compile=True` to compile optimizer steps on CPU or GPU.
The tests cover supported foreach optimizers, including Muon, AdaMuon, AdaGO, and NorMuon,
as well as Lion, native PyTorch AdamW, and StableAdamW:

```python
optimizer = create_optimizer(model, 'lion', lr=1e-3, foreach=False, compile=True)
```

The factory converts float learning rates to tensors on the model's device.
Schedulers can then change the learning rate without recompilation.
For native AdamW on CUDA, set `capturable=True` if you use a tensor rate with `foreach=True`.

Use `compile=False` (the default) for eager execution.
Use eager execution on Python 3.15 because PyTorch disables compilation for that version.
Pass `torch.compile()` options through `compile_kwargs`.
Run with `TORCH_LOGS=graph_breaks,recompiles` to inspect graph breaks and recompilation.

### Muon family

Muon, AdaMuon, AdaGO, and NorMuon accept `foreach=True` to batch momentum, weight decay,
and AdamW updates. Equal-shaped matrices on the same device and with the same dtype also
share a batched Newton-Schulz orthogonalization. Individual updates remain the default.
Set `foreach=True` or `foreach=None` to enable batching, or override it in a parameter group.

```python
optimizer = create_optimizer(model, 'muon', lr=0.02, foreach=True, compile=True)
```

The compiled path keeps state initialization, shape grouping, and step counters eager.
It passes changing learning rates and bias corrections as tensors to avoid recompilation.
The factory enables foreach automatically with `compile=True` unless you pass `foreach=False`.
For model-specific parameter grouping, construct the optimizer directly using its API example.
DistributedMuon retains its distributed update path.

Batching needs temporary storage for the stacked matrices. Low-precision batched or compiled
operations can round differently from individual updates. Measure optimizer updates and full
training steps on your CUDA workload using the [benchmark](benchmark.md).
Large matrix workloads can run slower with eager foreach; compilation can still improve their training speed.

## Discover components

Filter component names with wildcard patterns:

```python
from pytorch_optimizer import (
    get_supported_loss_functions,
    get_supported_lr_schedulers,
    get_supported_optimizers,
)

all_optimizers = get_supported_optimizers()
adam_family = get_supported_optimizers('adam*')
selected = get_supported_optimizers(['adam*', 'ranger*'])
cosine_schedulers = get_supported_lr_schedulers('cosine*')
focal_losses = get_supported_loss_functions('*focal*')
```

Check the [scheduler reference](lr_scheduler.md) and [loss reference](loss.md) for signatures and usage details.
Read the [FAQ](qa.md) before you train with Hessian-based optimizers such as AdaHessian and SophiaH.
