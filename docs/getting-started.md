# Getting Started

## Installation

Use Python 3.10 or later with PyTorch 2.1 or later. The package installer selects dependencies compatible with your Python version.

```bash
pip install pytorch-optimizer
```

Install optional integrations from their projects when you need them:
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
[Q-GaLore](https://github.com/VITA-Group/Q-GaLore), or
[TorchAO](https://github.com/pytorch/ao).

## Run a training step

Pass model parameters to an optimizer class, then use the PyTorch training loop:

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

Use `load_optimizer()` when you store the optimizer name in a training configuration:

```python
from pytorch_optimizer import load_optimizer

optimizer = load_optimizer('adamp')(model.parameters(), lr=1e-3)
```

You can also load an optimizer class through Torch Hub:

```python
optimizer_class = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
optimizer = optimizer_class(model.parameters(), lr=1e-3)
```

## Combine optimizer options

Use `create_optimizer()` to configure weight decay and wrappers with the model:

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

See the [optimizer reference](optimizer.md) for the available options and optimizer-specific arguments.

For fp16 parameters, `StableAdamW` stores second moments in fp32 to avoid underflow and overflow when squaring gradients.
This uses two additional bytes per parameter compared with fp16 second moments. Its bf16 second moments retain bf16 storage.

## Compile optimizer steps

Set `compile=True` to compile optimizer steps on CPU or GPU. We test Lion, native PyTorch AdamW, and StableAdamW:

```python
optimizer = create_optimizer(model, 'lion', lr=1e-3, foreach=False, compile=True)
```

The factory converts float learning rates to tensors on the model's device so schedulers can change them without recompilation.
For native AdamW on CUDA, set `capturable=True` if you use a tensor rate with `foreach=True`.

Use `compile=False` (the default) for eager execution.
Use eager execution on Python 3.15, where PyTorch disables compilation.
Pass `torch.compile()` options through `compile_kwargs`.
Run with `TORCH_LOGS=graph_breaks,recompiles` to inspect graph breaks and recompilation.

## Discover components

Filter component names with shell-style patterns:

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

Browse the [scheduler reference](lr_scheduler.md) and [loss reference](loss.md) for signatures and usage details.
For Hessian-based optimizers such as AdaHessian and SophiaH, read the [FAQ](qa.md) before training.
