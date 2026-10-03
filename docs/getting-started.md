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

## Compile optimizer steps

Compilation is optional and starts on the first step.
Lion, native PyTorch AdamW, and StableAdamW are tested with compilation:

Compiled steps work on CPU and GPU. The default Inductor backend generates C++ kernels on CPU and typically
Triton kernels on CUDA. A supported compiler toolchain is required for the selected backend.

```python
lr = torch.tensor(1e-3, device=next(model.parameters()).device)
optimizer = create_optimizer(model, 'lion', lr=lr, foreach=False, compile_step=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
```

Use `'adamw'` for native AdamW or `'stableadamw'` for StableAdamW.
Keep the usual `optimizer.step()` and `scheduler.step()` calls, in that order.
Eager execution remains the default. Set `compile_step=False` to use it.
`compile_kwargs` forwards options such as `backend`, `mode`, and `fullgraph` to `torch.compile()`.
Dynamic tracing is enabled by default so Python step counters do not force compilation at each step.

Use a scalar tensor learning rate when a scheduler changes it. Replacing a Python float rate can trigger
recompilation, depending on the PyTorch version. Tensor values can change without adding value guards.
Native AdamW with `foreach=True` requires `capturable=True` for a tensor rate; use this combination on CUDA.
Lion and StableAdamW accept tensor rates with either foreach setting.
StableAdamW keeps its RMS scale and adaptive step sizes on-device, avoiding scalar extraction in both paths.

The first steps may compile separate graphs for state initialization and steady training.
Changes to parameter shapes, gradient availability, or optimizer options can require new graphs.
Native AdamW inserts graph breaks around its gradient-mode wrapper; its tensor updates can still form one graph.
Closures and optional wrappers may introduce additional breaks.

Run training with `TORCH_LOGS=graph_breaks,recompiles` to inspect compilation decisions.
For a class constructed directly, keep an eager step and compile a separate callable after attaching any scheduler:

```python
from pytorch_optimizer import Lion

optimizer = Lion(model.parameters(), lr=lr, foreach=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
eager_step = optimizer.step
compiled_step = torch.compile(eager_step, dynamic=True)
```

Measure steady step time after warmup on the target GPU. Compilation startup cost can outweigh savings in short runs,
and performance depends on the installed compiler backend.

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
