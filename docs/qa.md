# Frequently asked questions

## Q1) SophiaH, AdaHessian optimizers give ```RuntimeError: ~ tensors does not require grad and does not have a grad_fn``` in `compute_hutchinson_hessian()`.

`create_graph` must be set `True` when calling `backward()`. here's [an example](https://github.com/kozistr/pytorch_optimizer/issues/194#issuecomment-1723167466).

## Q2) Memory leak happens when using SophiaH, AdaHessian optimizers.

`torch.autograd.grad` with complex gradient flows sometimes leads memory leak issues, and you might encounter OOM issue. [related issue](https://github.com/kozistr/pytorch_optimizer/issues/278)

## Q3) How to run visualizations?

Run `make visualize` or `python3 -m examples.visualize_optimizers` on the project root.
