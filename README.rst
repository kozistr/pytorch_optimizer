=================
pytorch-optimizer
=================

+--------------+------------------------------------------+
| Build        | |workflow| |Documentation Status|        |
+--------------+------------------------------------------+
| Quality      | |codecov| |black|                        |
+--------------+------------------------------------------+
| Package      | |PyPI version| |PyPI pyversions|         |
+--------------+------------------------------------------+
| Status       | |PyPi download| |PyPi month download|    |
+--------------+------------------------------------------+

| **pytorch-optimizer** is bunch of optimizer collections in PyTorch. Also, including useful optimization ideas.
| Most of the implementations are based on the original paper, but I added some tweaks.
| Highly inspired by `pytorch-optimizer <https://github.com/jettify/pytorch-optimizer>`__.

Documentation
-------------

https://pytorch-optimizers.readthedocs.io/en/latest/

Usage
-----

Install
~~~~~~~

::

    $ pip3 install -U pytorch-optimizer

or

::

    $ pip3 install -U --no-deps pytorch-optimizer

Simple Usage
~~~~~~~~~~~~

::

    from pytorch_optimizer import AdamP

    model = YourModel()
    optimizer = AdamP(model.parameters())

    # or you can use optimizer loader, simply passing a name of the optimizer.

    from pytorch_optimizer import load_optimizer

    model = YourModel()
    opt = load_optimizer(optimizer='adamp')
    optimizer = opt(model.parameters())

Also, you can load the optimizer via `torch.hub`

::

    import torch

    model = YourModel()
    opt = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
    optimizer = opt(model.parameters())


And you can check the supported optimizers & lr schedulers.

::

    from pytorch_optimizer import get_supported_optimizers, get_supported_lr_schedulers

    supported_optimizers = get_supported_optimizers()
    supported_lr_schedulers = get_supported_lr_schedulers()


Supported Optimizers
--------------------

+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Optimizer    | Description                                                                            | Official Code                                                                     | Paper                                                                                         |
+==============+========================================================================================+===================================================================================+===============================================================================================+
| AdaBelief    | *Adapting Step-sizes by the Belief in Observed Gradients*                              | `github <https://github.com/juntang-zhuang/Adabelief-Optimizer>`__                | `https://arxiv.org/abs/2010.07468 <https://arxiv.org/abs/2010.07468>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| AdaBound     | *Adaptive Gradient Methods with Dynamic Bound of Learning Rate*                        | `github <https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py>`__   | `https://openreview.net/forum?id=Bkg3g2R9FX <https://openreview.net/forum?id=Bkg3g2R9FX>`__   |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| AdaHessian   | *An Adaptive Second Order Optimizer for Machine Learning*                              | `github <https://github.com/amirgholami/adahessian>`__                            | `https://arxiv.org/abs/2006.00719 <https://arxiv.org/abs/2006.00719>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| AdamD        | *Improved bias-correction in Adam*                                                     |                                                                                   | `https://arxiv.org/abs/2110.10828 <https://arxiv.org/abs/2110.10828>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| AdamP        | *Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights*         | `github <https://github.com/clovaai/AdamP>`__                                     | `https://arxiv.org/abs/2006.08217 <https://arxiv.org/abs/2006.08217>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| diffGrad     | *An Optimization Method for Convolutional Neural Networks*                             | `github <https://github.com/shivram1987/diffGrad>`__                              | `https://arxiv.org/abs/1909.11015v3 <https://arxiv.org/abs/1909.11015v3>`__                   |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| MADGRAD      | *A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic*               | `github <https://github.com/facebookresearch/madgrad>`__                          | `https://arxiv.org/abs/2101.11075 <https://arxiv.org/abs/2101.11075>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| RAdam        | *On the Variance of the Adaptive Learning Rate and Beyond*                             | `github <https://github.com/LiyuanLucasLiu/RAdam>`__                              | `https://arxiv.org/abs/1908.03265 <https://arxiv.org/abs/1908.03265>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Ranger       | *a synergistic optimizer combining RAdam and LookAhead, and now GC in one optimizer*   | `github <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer>`__          | `https://bit.ly/3zyspC3 <https://bit.ly/3zyspC3>`__                                           |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Ranger21     | *a synergistic deep learning optimizer*                                                | `github <https://github.com/lessw2020/Ranger21>`__                                | `https://arxiv.org/abs/2106.13731 <https://arxiv.org/abs/2106.13731>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Lamb         | *Large Batch Optimization for Deep Learning*                                           | `github <https://github.com/cybertronai/pytorch-lamb>`__                          | `https://arxiv.org/abs/1904.00962 <https://arxiv.org/abs/1904.00962>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Shampoo      | *Preconditioned Stochastic Tensor Optimization*                                        | `github <https://github.com/moskomule/shampoo.pytorch>`__                         | `https://arxiv.org/abs/1802.09568 <https://arxiv.org/abs/1802.09568>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Nero         | *Learning by Turning: Neural Architecture Aware Optimisation*                          | `github <https://github.com/jxbz/nero>`__                                         | `https://arxiv.org/abs/2102.07227 <https://arxiv.org/abs/2102.07227>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Adan         | *Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models*               | `github <https://github.com/sail-sg/Adan>`__                                      | `https://arxiv.org/abs/2208.06677 <https://arxiv.org/abs/2208.06677>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Adai         | *Disentangling the Effects of Adaptive Learning Rate and Momentum*                     | `github <https://github.com/zeke-xie/adaptive-inertia-adai>`__                    | `https://arxiv.org/abs/2006.15815 <https://arxiv.org/abs/2006.15815>`__                       |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| GSAM         | *Surrogate Gap Guided Sharpness-Aware Minimization*                                    | `github <https://github.com/juntang-zhuang/GSAM>`__                               | `https://openreview.net/pdf?id=edONMAnhLu- <https://openreview.net/pdf?id=edONMAnhLu->`__     |
+--------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+

Useful Resources
----------------

Several optimization ideas to regularize & stabilize the training. Most
of the ideas are applied in ``Ranger21`` optimizer.

Also, most of the captures are taken from ``Ranger21`` paper.

+------------------------------------------+---------------------------------------------+--------------------------------------------+
| `Adaptive Gradient Clipping`_            | `Gradient Centralization`_                  | `Softplus Transformation`_                 |
+------------------------------------------+---------------------------------------------+--------------------------------------------+
| `Gradient Normalization`_                | `Norm Loss`_                                | `Positive-Negative Momentum`_              |
+------------------------------------------+---------------------------------------------+--------------------------------------------+
| `Linear learning rate warmup`_           | `Stable weight decay`_                      | `Explore-exploit learning rate schedule`_  |
+------------------------------------------+---------------------------------------------+--------------------------------------------+
| `Lookahead`_                             | `Chebyshev learning rate schedule`_         | `(Adaptive) Sharpness-Aware Minimization`_ |
+------------------------------------------+---------------------------------------------+--------------------------------------------+
| `On the Convergence of Adam and Beyond`_ | `Gradient Surgery for Multi-Task Learning`_ |                                            |
+------------------------------------------+---------------------------------------------+--------------------------------------------+

Adaptive Gradient Clipping
--------------------------

| This idea originally proposed in ``NFNet (Normalized-Free Network)`` paper.
| ``AGC (Adaptive Gradient Clipping)`` clips gradients based on the ``unit-wise ratio of gradient norms to parameter norms``.

-  code : `github <https://github.com/deepmind/deepmind-research/tree/master/nfnets>`__
-  paper : `arXiv <https://arxiv.org/abs/2102.06171>`__

Gradient Centralization
-----------------------

+-----------------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/gradient_centralization.png  |
+-----------------------------------------------------------------------------------------------------------------+

``Gradient Centralization (GC)`` operates directly on gradients by centralizing the gradient to have zero mean.

-  code : `github <https://github.com/Yonghongwei/Gradient-Centralization>`__
-  paper : `arXiv <https://arxiv.org/abs/2004.01461>`__

Softplus Transformation
-----------------------

By running the final variance denom through the softplus function, it lifts extremely tiny values to keep them viable.

-  paper : `arXiv <https://arxiv.org/abs/1908.00700>`__

Gradient Normalization
----------------------

Norm Loss
---------

+---------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/norm_loss.png  |
+---------------------------------------------------------------------------------------------------+

-  paper : `arXiv <https://arxiv.org/abs/2103.06583>`__

Positive-Negative Momentum
--------------------------

+--------------------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/positive_negative_momentum.png  |
+--------------------------------------------------------------------------------------------------------------------+

-  code : `github <https://github.com/zeke-xie/Positive-Negative-Momentum>`__
-  paper : `arXiv <https://arxiv.org/abs/2103.17182>`__

Linear learning rate warmup
---------------------------

+----------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/linear_lr_warmup.png  |
+----------------------------------------------------------------------------------------------------------+

-  paper : `arXiv <https://arxiv.org/abs/1910.04209>`__

Stable weight decay
-------------------

+-------------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/stable_weight_decay.png  |
+-------------------------------------------------------------------------------------------------------------+

-  code : `github <https://github.com/zeke-xie/stable-weight-decay-regularization>`__
-  paper : `arXiv <https://arxiv.org/abs/2011.11152>`__

Explore-exploit learning rate schedule
--------------------------------------

+---------------------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/explore_exploit_lr_schedule.png  |
+---------------------------------------------------------------------------------------------------------------------+

-  code : `github <https://github.com/nikhil-iyer-97/wide-minima-density-hypothesis>`__
-  paper : `arXiv <https://arxiv.org/abs/2003.03977>`__

Lookahead
---------

| ``k`` steps forward, 1 step back. ``Lookahead`` consisting of keeping an exponential moving average of the weights that is
| updated and substituted to the current weights every ``k_{lookahead}`` steps (5 by default).

-  code : `github <https://github.com/alphadl/lookahead.pytorch>`__
-  paper : `arXiv <https://arxiv.org/abs/1907.08610v2>`__

Chebyshev learning rate schedule
--------------------------------

Acceleration via Fractal Learning Rate Schedules

-  paper : `arXiv <https://arxiv.org/abs/2103.01338v1>`__

(Adaptive) Sharpness-Aware Minimization
---------------------------------------

| Sharpness-Aware Minimization (SAM) simultaneously minimizes loss value and loss sharpness.
| In particular, it seeks parameters that lie in neighborhoods having uniformly low loss.

-  SAM paper : `paper <https://arxiv.org/abs/2010.01412>`__
-  ASAM paper : `paper <https://arxiv.org/abs/2102.11600>`__
-  A/SAM code : `github <https://github.com/davda54/sam>`__

On the Convergence of Adam and Beyond
-------------------------------------

- paper : `paper <https://openreview.net/forum?id=ryQu7f-RZ>`__

Gradient Surgery for Multi-Task Learning
----------------------------------------

- paper : `paper <https://arxiv.org/abs/2001.06782>`__

Citations
---------

`AdamP <https://github.com/clovaai/AdamP#how-to-cite>`__

`Adaptive Gradient Clipping <https://ui.adsabs.harvard.edu/abs/2021arXiv210206171B/exportcitation>`__

`Chebyshev LR Schedules <https://ui.adsabs.harvard.edu/abs/2021arXiv210301338A/exportcitation>`__

`Gradient Centralization <https://github.com/Yonghongwei/Gradient-Centralization#citation>`__

`Lookahead <https://ui.adsabs.harvard.edu/abs/2019arXiv190708610Z/exportcitation>`__

`RAdam <https://github.com/LiyuanLucasLiu/RAdam#citation>`__

`Norm Loss <https://ui.adsabs.harvard.edu/abs/2021arXiv210306583G/exportcitation>`__

`Positive-Negative Momentum <https://github.com/zeke-xie/Positive-Negative-Momentum#citing>`__

`Explore-Exploit Learning Rate Schedule <https://ui.adsabs.harvard.edu/abs/2020arXiv200303977I/exportcitation>`__

`On the adequacy of untuned warmup for adaptive optimization <https://ui.adsabs.harvard.edu/abs/2019arXiv191004209M/exportcitation>`__

`Stable weight decay regularization <https://github.com/zeke-xie/stable-weight-decay-regularization#citing>`__

`Softplus transformation <https://ui.adsabs.harvard.edu/abs/2019arXiv190800700T/exportcitation>`__

`MADGRAD <https://github.com/facebookresearch/madgrad#tech-report>`__

`AdaHessian <https://github.com/amirgholami/adahessian#citation>`__

`AdaBound <https://github.com/Luolc/AdaBound#citing>`__

`Adabelief <https://ui.adsabs.harvard.edu/abs/2020arXiv201007468Z/exportcitation>`__

`Sharpness-aware minimization <https://ui.adsabs.harvard.edu/abs/2020arXiv201001412F/exportcitation>`__

`Adaptive Sharpness-aware minimization <https://ui.adsabs.harvard.edu/abs/2021arXiv210211600K/exportcitation>`__

`diffGrad <https://ui.adsabs.harvard.edu/abs/2019arXiv190911015D/exportcitation>`__

`On the Convergence of Adam and Beyond <https://ui.adsabs.harvard.edu/abs/2019arXiv190409237R/exportcitation>`__

`Gradient surgery for multi-task learning <https://ui.adsabs.harvard.edu/abs/2020arXiv200106782Y/exportcitation>`__

`AdamD <https://ui.adsabs.harvard.edu/abs/2021arXiv211010828S/exportcitation>`__

`Shampoo <https://ui.adsabs.harvard.edu/abs/2018arXiv180209568G/exportcitation>`__

`Nero <https://ui.adsabs.harvard.edu/abs/2021arXiv210207227L/exportcitation>`__

`Adan <https://ui.adsabs.harvard.edu/abs/2022arXiv220806677X/exportcitation>`__

`Adai <https://github.com/zeke-xie/adaptive-inertia-adai#citing>`__

`GSAM <https://github.com/juntang-zhuang/GSAM#citation>`__

Citation
--------

Please cite original authors of optimization algorithms. If you use this software, please cite it as below.
Or you can get from "cite this repository" button.

::

    @software{Kim_pytorch_optimizer_Bunch_of_2022,
        author = {Kim, Hyeongchan},
        month = {1},
        title = {{pytorch_optimizer: Bunch of optimizer implementations in PyTorch with clean-code, strict types}},
        version = {1.0.0},
        year = {2022}
    }

Author
------

Hyeongchan Kim / `@kozistr <http://kozistr.tech/about>`__

.. |workflow| image:: https://github.com/kozistr/pytorch_optimizer/actions/workflows/ci.yml/badge.svg?branch=main
.. |Documentation Status| image:: https://readthedocs.org/projects/pytorch-optimizers/badge/?version=latest
   :target: https://pytorch-optimizers.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://badge.fury.io/py/pytorch-optimizer.svg
   :target: https://badge.fury.io/py/pytorch-optimizer
.. |PyPi download| image:: https://pepy.tech/badge/pytorch-optimizer
   :target: https://pepy.tech/project/pytorch-optimizer
.. |PyPi month download| image:: https://pepy.tech/badge/pytorch-optimizer/month
   :target: https://pepy.tech/project/pytorch-optimizer
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/pytorch-optimizer.svg
   :target: https://pypi.python.org/pypi/pytorch-optimizer/
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. |codecov| image:: https://codecov.io/gh/kozistr/pytorch_optimizer/branch/main/graph/badge.svg?token=L4K00EA0VD
   :target: https://codecov.io/gh/kozistr/pytorch_optimizer
