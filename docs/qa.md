# Frequently asked questions

## Q1) SophiaH, AdaHessian optimizers give ```RuntimeError: ~ tensors does not require grad and does not have a grad_fn``` in `compute_hutchinson_hessian()`.

`create_graph` must be set `True` when calling `backward()`. here's [an example](https://github.com/kozistr/pytorch_optimizer/issues/194#issuecomment-1723167466).
