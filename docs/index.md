# pytorch-optimizer

Choose from more than 100 optimizers, plus learning rate schedulers and loss functions for PyTorch.
Use an optimizer class or select an optimizer by name.
Use `create_optimizer()` to configure weight decay, gradient centralization, and Lookahead.

```python
from torch import nn
from pytorch_optimizer import create_optimizer

model = nn.Linear(10, 1)
optimizer = create_optimizer(model, 'adamp', lr=1e-3)
```

## Start training

Follow the [getting started guide](getting-started.md) to install the package and run a training step.
The guide also shows how to find component names.
Read the [supported algorithms](https://github.com/kozistr/pytorch_optimizer#supported-optimizers) for descriptions and
links to papers.

## API reference

| Reference | Contents |
| --- | --- |
| [Optimizers](optimizer.md) | Optimizer classes, loaders, and parameter groups |
| [Learning rate schedulers](lr_scheduler.md) | Warmup, cosine, polynomial, and other schedules |
| [Loss functions](loss.md) | Classification and segmentation objectives |
| [Utilities](util.md) | Component discovery, gradient operations, and CPU offloading |
| [Base optimizer](base.md) | Shared validation and update helpers for optimizer authors |

## Compare and troubleshoot

- Compare execution time and memory use in the [optimizer benchmarks](benchmark.md).
- Compare optimizer paths in the [visualizations](visualization.md).
- Check the [FAQ](qa.md) for Hessian computation and memory issues.
- Read the [changelog](changelogs/index.md) for release notes.

## Contribute

Follow the [contributing guide](https://github.com/kozistr/pytorch_optimizer/blob/main/CONTRIBUTING.md) to add an
optimizer, revise documentation, or report a bug.
Check the [license notes](https://github.com/kozistr/pytorch_optimizer#license-notes) before you use code with
additional terms.
