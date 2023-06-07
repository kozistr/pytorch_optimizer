## Change Log

### Feature

* Implement Amos optimizer (#174)
  * [An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale](https://arxiv.org/abs/2210.11693)
* Implement SingSGD optimizer (#176) (thanks to @i404788)
  * [Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
* Implement AdaHessian optimizer (#176) (thanks to @i404788)
  * [An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719)
* Implement SophiaH optimizer (#176) (thanks to @i404788)
  * [A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) 
* Implement re-usable tools to compute hessian in `BaseOptimizer` (#176)

### Diff

[2.9.1...2.10.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.9.1...v2.10.0)
