# pytorch-optimizer

Use more than 100 optimizers, learning rate schedulers, and loss functions with PyTorch.
Start with a named optimizer, or use `create_optimizer()` to combine weight decay, gradient centralization, and Lookahead.

```python
from torch import nn
from pytorch_optimizer import create_optimizer

model = nn.Linear(10, 1)
optimizer = create_optimizer(model, 'adamp', lr=1e-3)
```

## Start training

Follow the [getting started guide](getting-started.md) for installation, a training step, and component discovery.
For algorithm descriptions and links to papers, browse the
[supported algorithms](https://github.com/kozistr/pytorch_optimizer#supported-optimizers).

## API reference

| Reference | Contents |
| --- | --- |
| [Optimizers](optimizer.md) | Optimizer classes, loaders, and parameter groups |
| [Learning rate schedulers](lr_scheduler.md) | Warmup, cosine, polynomial, and other schedules |
| [Loss functions](loss.md) | Classification and segmentation objectives |
| [Utilities](util.md) | Component discovery, gradient operations, and CPU offloading |
| [Base optimizer](base.md) | Shared validation and update helpers for optimizer authors |

## Compare and troubleshoot

- Compare runtime and memory in the [foreach benchmarks](benchmark.md).
- Inspect optimizer paths in the [visualizations](visualization.md).
- Check the [FAQ](qa.md) for Hessian computation and memory issues.
- Read the [changelog](changelogs/index.md) for release notes.

## Contribute

See the [contributing guide](https://github.com/kozistr/pytorch_optimizer/blob/main/CONTRIBUTING.md)
to add an optimizer, improve the docs, or report a bug.
Check the [license notes](https://github.com/kozistr/pytorch_optimizer#license-notes) before using code with additional terms.
