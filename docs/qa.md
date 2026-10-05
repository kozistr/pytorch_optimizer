# Frequently Asked Questions

## Hessian computation fails with a gradient error

SophiaH and AdaHessian need the gradient graph to compute the Hessian.
Pass `create_graph=True` to `backward()` if `compute_hutchinson_hessian()` reports that tensors do not require
gradients:

```python
loss.backward(create_graph=True)
```

See the [usage example](https://github.com/kozistr/pytorch_optimizer/issues/194#issuecomment-1723167466).

## Memory usage grows with Hessian-based optimizers

Retaining gradient graphs for SophiaH or AdaHessian can increase memory use and cause out-of-memory errors.
Read the [memory usage discussion](https://github.com/kozistr/pytorch_optimizer/issues/278) for reported cases.

## Run optimizer visualizations

Run `uv run just visualize` from the repository root after you install the plotting dependencies.
You can also run `uv run python -m examples.visualize_optimizers`.
Follow the [visualization guide](visualization.md) for setup instructions and plots.
