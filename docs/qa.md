# Frequently Asked Questions

## Hessian computation fails with a gradient error

SophiaH and AdaHessian need the gradient graph to compute the Hessian. If `compute_hutchinson_hessian()` reports that tensors do not require gradients, pass `create_graph=True` to `backward()`:

```python
loss.backward(create_graph=True)
```

See the [usage example](https://github.com/kozistr/pytorch_optimizer/issues/194#issuecomment-1723167466).

## Memory usage grows with Hessian-based optimizers

When using SophiaH or AdaHessian, retaining gradient graphs can increase memory usage and cause out-of-memory errors.
See the [memory usage discussion](https://github.com/kozistr/pytorch_optimizer/issues/278) for reported cases.

## Run optimizer visualizations

Run `uv run just visualize` or `uv run python -m examples.visualize_optimizers` from the repository root.
See the [visualization gallery](visualization.md) for the generated plots.
