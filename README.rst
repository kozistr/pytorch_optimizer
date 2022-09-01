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

    $ pip3 install pytorch-optimizer

Simple Usage
~~~~~~~~~~~~

::

    from pytorch_optimizer import AdamP

    ...
    model = YourModel()
    optimizer = AdamP(model.parameters())
    ...

or you can use optimizer loader, simply passing a name of the optimizer.

::

    from pytorch_optimizer import load_optimizer

    ...
    model = YourModel()
    opt = load_optimizer(optimizer='adamp')
    optimizer = opt(model.parameters())
    ...

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

`AdamP <https://scholar.googleusercontent.com/scholar.bib?q=info:SfSq5UFS71wJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YevydU:AAGBfm0AAAAAYxCp0dVqrS10vvLfEDcY31SdH8ZRpeB4&scisig=AAGBfm0AAAAAYxCp0bLEn4nNd2Gmpb64J-nsN62Hq19N&scisf=4&ct=citation&cd=-1&hl=en>`__

`Adaptive Gradient Clipping (AGC) <https://scholar.googleusercontent.com/scholar.bib?q=info:G6OwKvfrhU4J:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YesC_0:AAGBfm0AAAAAYxCqE_3u1oAcHorMaAJ_SR7Xo5PvdxIC&scisig=AAGBfm0AAAAAYxCqEz7D8y15Q5sJL5QUdbpTMdFHGSMi&scisf=4&ct=citation&cd=-1&hl=en>`__

`Chebyshev LR Schedules <https://scholar.googleusercontent.com/scholar.bib?q=info:5bxSTRao5pUJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YesV7g:AAGBfm0AAAAAYxCqT7jEP6cOz39vHjSXD71OiD_WHNeu&scisig=AAGBfm0AAAAAYxCqTxBAT7yBvhGW1KZopv6tYDL6fjhq&scisf=4&ct=citation&cd=-1&hl=en>`__

`Gradient Centralization (GC) <https://scholar.googleusercontent.com/scholar.bib?q=info:MQDRtwz4RekJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeskLw:AAGBfm0AAAAAYxCqiLx6z7Lo-Fag54T6c22UyMxC3uKU&scisig=AAGBfm0AAAAAYxCqiDzweYqjl8tPPjAVYv4y42-amW04&scisf=4&ct=citation&cd=-1&hl=en>`__

`Lookahead <https://scholar.googleusercontent.com/scholar.bib?q=info:A1J2Cn9LEyQJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yest68:AAGBfm0AAAAAYxCqr68LW2mC6SXXXXIEv17IH1VfVwTU&scisig=AAGBfm0AAAAAYxCqr0ZQGEPcASa4BcFlRIMYfC_ELoH3&scisf=4&ct=citation&cd=-1&hl=en>`__

`RAdam <https://scholar.googleusercontent.com/scholar.bib?q=info:tTLLKZi0NB4J:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yes-Kc:AAGBfm0AAAAAYxCq4KdbtBaCrCnPM3teTRbkG2ke4zu1&scisig=AAGBfm0AAAAAYxCq4DKANM54ZoMqj8sYTKjhrrWTYZJv&scisf=4&ct=citation&cd=-1&hl=en>`__

`Norm Loss <https://scholar.googleusercontent.com/scholar.bib?q=info:cgudi9fC610J:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YetGG8:AAGBfm0AAAAAYxCrAG8mPyX5faDy-Orn0sNT3laCqhCX&scisig=AAGBfm0AAAAAYxCrAPhudmT6SGj0XyHAGuBIgn4iP9UM&scisf=4&ct=citation&cd=-1&hl=en>`__

`Positive-Negative Momentum <https://scholar.googleusercontent.com/scholar.bib?q=info:EU4LbWCU44UJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YetNIE:AAGBfm0AAAAAYxCrLIFD4YhCP2b755xkmgM9ekT5z2I3&scisig=AAGBfm0AAAAAYxCrLA0s6cI4xGBVGFOpGDBJkD4jW45M&scisf=4&ct=citation&cd=-1&hl=en>`__

`Explore-Exploit Learning Rate Schedule <https://scholar.googleusercontent.com/scholar.bib?q=info:-Z0_Ot7wtzsJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YetRPU:AAGBfm0AAAAAYxCrXPVjSJKqfwDN1V1KDkX--4xZuQ3d&scisig=AAGBfm0AAAAAYxCrXLMftLTqnC4BUjTH8TEDoeg8Xn0P&scisf=4&ct=citation&cd=-1&hl=en>`__

`On the adequacy of untuned warmup for adaptive optimization <https://scholar.googleusercontent.com/scholar.bib?q=info:_xl7KQ5GS8wJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yetb_s:AAGBfm0AAAAAYxCrd_t2aLAHKkunOI588UJkaMygzX7V&scisig=AAGBfm0AAAAAYxCrd4xDt7wmBQYV2J88Dv1klVIEEldW&scisf=4&ct=citation&cd=-1&hl=en>`__

`Stable weight decay regularization <https://scholar.googleusercontent.com/scholar.bib?q=info:braJqOHCLpcJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yetu34:AAGBfm0AAAAAYxCro36JSgGOwWVwx8K21_sJaiJCi_tc&scisig=AAGBfm0AAAAAYxCro42f96rMxskixD8vZdyLuRCv9hzp&scisf=4&ct=citation&cd=-1&hl=en>`__

`Softplus transformation <https://scholar.googleusercontent.com/scholar.bib?q=info:_V_Tt16gXUsJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yet3gY:AAGBfm0AAAAAYxCrxgbrSUaRQqStYNBuVBPS3TMRgH7f&scisig=AAGBfm0AAAAAYxCrxqnu8UQn70pqZWxbBoJaz05eCgsj&scisf=4&ct=citation&cd=-1&hl=en>`__

`MADGRAD <https://scholar.googleusercontent.com/scholar.bib?q=info:WnYNAExj8yEJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yet6g8:AAGBfm0AAAAAYxCr8g-OAPHACQZtBVamCAXY3mUPO7qR&scisig=AAGBfm0AAAAAYxCr8iVTWljaTOsxZ9ZHce61Uh5rYWdB&scisf=4&ct=citation&cd=-1&hl=en>`__

`AdaHessian <https://scholar.googleusercontent.com/scholar.bib?q=info:NVTf2oQp6YoJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeqDj8:AAGBfm0AAAAAYxCsFj89NAaxz72Tc2BaFva6FGFHuzjO&scisig=AAGBfm0AAAAAYxCsFm7SeFVY6NaIy5w0BOLAVGM4oy-z&scisf=4&ct=citation&cd=-1&hl=en>`__

`AdaBound <https://scholar.googleusercontent.com/scholar.bib?q=info:CsrDHbimhWgJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeqXZQ:AAGBfm0AAAAAYxCsRZR-WfagzOhOzHZ3ARAlehesAaQL&scisig=AAGBfm0AAAAAYxCsRSRkCJhTl9QisH1o5k8cbHBOOaQ0&scisf=4&ct=citation&cd=-1&hl=en>`__

`Adabelief <https://scholar.googleusercontent.com/scholar.bib?q=info:cf1gkNMQCAsJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeqcPk:AAGBfm0AAAAAYxCsaPn6O2pgnuIZmWlssnrLY7Zug1ab&scisig=AAGBfm0AAAAAYxCsaPiac1Ktzqa7-8wabbO3pQzq2ezC&scisf=4&ct=citation&cd=-1&hl=en>`__

`Sharpness-aware minimization <https://scholar.googleusercontent.com/scholar.bib?q=info:621rS0TnyooJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeqkcY:AAGBfm0AAAAAYxCsicYP7tw5aRNOjjXwkA4Vow-7jzWX&scisig=AAGBfm0AAAAAYxCsibGf462P1_gsWErL-yeGdIeNHywO&scisf=4&ct=citation&cd=-1&hl=en>`__

`Adaptive Sharpness-aware minimization <https://scholar.googleusercontent.com/scholar.bib?q=info:ta4j_XtLqXYJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YeqhhE:AAGBfm0AAAAAYxCsnhEGLjlU7PCikAYnM6LYuACuKcfu&scisig=AAGBfm0AAAAAYxCsno-VG_RWK0tOtqZdWxel6qTKtNyC&scisf=4&ct=citation&cd=-1&hl=en>`__

`diffGrad <https://scholar.googleusercontent.com/scholar.bib?q=info:yGmD33AMjN4J:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yeqp7I:AAGBfm0AAAAAYxCsv7IYbE3ozFQrbhjAxbBdhbcNrNaT&scisig=AAGBfm0AAAAAYxCsv2mDmsNyW0R1koLK3vG04K7HEyRW&scisf=4&ct=citation&cd=-1&hl=en>`__

`On the Convergence of Adam and Beyond <https://scholar.googleusercontent.com/scholar.bib?q=info:B0s07Z6wFWkJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yeq5VI:AAGBfm0AAAAAYxCs_VIET-w3Fc6Bx3B7pbnercaue84a&scisig=AAGBfm0AAAAAYxCs_Rzcu3G4tmMrxOdaeXsfN9RSp3aA&scisf=4&ct=citation&cd=-1&hl=en>`__

`Gradient surgery for multi-task learning <https://scholar.googleusercontent.com/scholar.bib?q=info:ae9CdgI_CtkJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YerBWY:AAGBfm0AAAAAYxCtHWZzzktUQ2GRhrSx_LWh7AiWbeUV&scisig=AAGBfm0AAAAAYxCtHaXMBqe9K0CCS9McXDPM8BRHHrTD&scisf=4&ct=citation&cd=-1&hl=en>`__

`AdamD <https://scholar.googleusercontent.com/scholar.bib?q=info:XimgvO50x1AJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YerIAo:AAGBfm0AAAAAYxCtOAq69M6dSH0RQEVyiQYk-5ToDCvA&scisig=AAGBfm0AAAAAYxCtOJRDGw1cq6WRv2NODkLE5sgxAPz-&scisf=4&ct=citation&cd=-1&hl=en>`__

`Shampoo <https://scholar.googleusercontent.com/scholar.bib?q=info:GQn55DATO9sJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YerS64:AAGBfm0AAAAAYxCtU65eO2d2kyAf36X-vcVbovISPAY9&scisig=AAGBfm0AAAAAYxCtUwoUqdIzjjuqat7lPKZylm3bO6io&scisf=4&ct=citation&cd=-1&hl=en>`__

`Nero <https://scholar.googleusercontent.com/scholar.bib?q=info:X7-f1Z-47X8J:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0Yercz8:AAGBfm0AAAAAYxCtaz9tFLHi82julKp6XCCGPZLRN2Qt&scisig=AAGBfm0AAAAAYxCta7MAiMjXj8qzcM8XBLi2AxsgVHIB&scisf=4&ct=citation&cd=-1&hl=en>`__

`Adan <https://scholar.googleusercontent.com/scholar.bib?q=info:rMUXKCk35EAJ:scholar.google.com/&output=citation&scisdr=CgX1Wk9EELXN0YerkVs:AAGBfm0AAAAAYxCtiVs7M7Oh9VkEVan-wY3IXOKyQtx1&scisig=AAGBfm0AAAAAYxCtiYyoEigNiau7MNmGcvqAEC8nSm-L&scisf=4&ct=citation&cd=-1&hl=en>`__

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
