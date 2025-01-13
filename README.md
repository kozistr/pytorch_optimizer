# pytorch-optimizer

|         |                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Build   | ![workflow](https://github.com/kozistr/pytorch_optimizer/actions/workflows/ci.yml/badge.svg?branch=main) [![Documentation Status](https://readthedocs.org/projects/pytorch-optimizers/badge/?version=latest)](https://pytorch-optimizers.readthedocs.io/en/latest/?badge=latest)                                                                                                                      |
| Quality | [![codecov](https://codecov.io/gh/kozistr/pytorch_optimizer/branch/main/graph/badge.svg?token=L4K00EA0VD)](https://codecov.io/gh/kozistr/pytorch_optimizer) ![black](https://img.shields.io/badge/code%20style-black-000000.svg) [![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff) |
| Package | [![PyPI version](https://badge.fury.io/py/pytorch-optimizer.svg)](https://badge.fury.io/py/pytorch-optimizer) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/pytorch-optimizer.svg)](https://pypi.python.org/pypi/pytorch-optimizer/)                                                                                                                                                     |
| Status  | [![PyPi download](https://static.pepy.tech/badge/pytorch-optimizer)](https://pepy.tech/project/pytorch-optimizer) [![PyPi month download](https://static.pepy.tech/badge/pytorch-optimizer/month)](https://pepy.tech/project/pytorch-optimizer)                                                                                                                                                       |
| License | [![apache](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)                                                                                                                                                                                                                                                                                     |

## The reasons why you use `pytorch-optimizer`.

* Wide range of supported optimizers. Currently, **89 optimizers (+ `bitsandbytes`, `qgalore`, `torchao`)**, **16 lr schedulers**, and **13 loss functions** are supported!
* Including many variants such as `Cautious`, `AdamD`, `Gradient Centrailiaztion`
* Easy to use, clean, and tested codes
* Active maintenance
* Somewhat a bit more optimized compared to the original implementation

Highly inspired by [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer).

## Getting Started

For more, see the [documentation](https://pytorch-optimizers.readthedocs.io/en/latest/).

Most optimizers are under MIT or Apache 2.0 license, but a few optimizers like `Fromage`, `Nero` have `CC BY-NC-SA 4.0 license`, which is non-commercial. 
So, please double-check the license before using it at your work.

### Installation

```bash
$ pip3 install pytorch-optimizer
```

From `v2.12.0`, `v3.1.0`, you can use `bitsandbytes`, `q-galore-torch`, `torchao` optimizers respectively!
please check [the bnb requirements](https://github.com/TimDettmers/bitsandbytes?tab=readme-ov-file#tldr), [q-galore-torch installation](https://github.com/VITA-Group/Q-GaLore?tab=readme-ov-file#install-q-galore-optimizer), [torchao installation](https://github.com/pytorch/ao?tab=readme-ov-file#installation)
 before installing it.

From `v3.0.0`, drop `Python 3.7` support. However, you can still use this package with `Python 3.7` by installing with `--ignore-requires-python` option.

### Simple Usage

```python
from pytorch_optimizer import AdamP

model = YourModel()
optimizer = AdamP(model.parameters())

# or you can use optimizer loader, simply passing a name of the optimizer.

from pytorch_optimizer import load_optimizer

optimizer = load_optimizer(optimizer='adamp')(model.parameters())

# if you install `bitsandbytes` optimizer, you can use `8-bit` optimizers from `pytorch-optimizer`.

optimizer = load_optimizer(optimizer='bnb_adamw8bit')(model.parameters())
```

Also, you can load the optimizer via `torch.hub`.

```python
import torch

model = YourModel()

opt = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
optimizer = opt(model.parameters())
```

If you want to build the optimizer with parameters & configs, there's `create_optimizer()` API.

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

## Supported Optimizers

You can check the supported optimizers with below code.

```python
from pytorch_optimizer import get_supported_optimizers

supported_optimizers = get_supported_optimizers()
```

or you can also search them with the filter(s).

```python
from pytorch_optimizer import get_supported_optimizers

get_supported_optimizers('adam*')
# ['adamax', 'adamg', 'adammini', 'adamod', 'adamp', 'adams', 'adamw']

get_supported_optimizers(['adam*', 'ranger*'])
# ['adamax', 'adamg', 'adammini', 'adamod', 'adamp', 'adams', 'adamw', 'ranger', 'ranger21']
```

| Optimizer     | Description                                                                                       | Official Code                                                                                                  | Paper                                                                                       | Citation                                                                                                                            |
|---------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| AdaBelief     | *Adapting Step-sizes by the Belief in Observed Gradients*                                         | [github](https://github.com/juntang-zhuang/Adabelief-Optimizer)                                                | <https://arxiv.org/abs/2010.07468>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv201007468Z/exportcitation)                                                        |
| AdaBound      | *Adaptive Gradient Methods with Dynamic Bound of Learning Rate*                                   | [github](https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py)                                   | <https://openreview.net/forum?id=Bkg3g2R9FX>                                                | [cite](https://github.com/Luolc/AdaBound#citing)                                                                                    |
| AdaHessian    | *An Adaptive Second Order Optimizer for Machine Learning*                                         | [github](https://github.com/amirgholami/adahessian)                                                            | <https://arxiv.org/abs/2006.00719>                                                          | [cite](https://github.com/amirgholami/adahessian#citation)                                                                          |
| AdamD         | *Improved bias-correction in Adam*                                                                |                                                                                                                | <https://arxiv.org/abs/2110.10828>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv211010828S/exportcitation)                                                        |
| AdamP         | *Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights*                    | [github](https://github.com/clovaai/AdamP)                                                                     | <https://arxiv.org/abs/2006.08217>                                                          | [cite](https://github.com/clovaai/AdamP#how-to-cite)                                                                                |
| diffGrad      | *An Optimization Method for Convolutional Neural Networks*                                        | [github](https://github.com/shivram1987/diffGrad)                                                              | <https://arxiv.org/abs/1909.11015v3>                                                        | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190911015D/exportcitation)                                                        |
| MADGRAD       | *A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic*                          | [github](https://github.com/facebookresearch/madgrad)                                                          | <https://arxiv.org/abs/2101.11075>                                                          | [cite](https://github.com/facebookresearch/madgrad#tech-report)                                                                     |
| RAdam         | *On the Variance of the Adaptive Learning Rate and Beyond*                                        | [github](https://github.com/LiyuanLucasLiu/RAdam)                                                              | <https://arxiv.org/abs/1908.03265>                                                          | [cite](https://github.com/LiyuanLucasLiu/RAdam#citation)                                                                            |
| Ranger        | *a synergistic optimizer combining RAdam and LookAhead, and now GC in one optimizer*              | [github](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)                                          | <https://bit.ly/3zyspC3>                                                                    | [cite](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer#citing-this-work)                                                |
| Ranger21      | *a synergistic deep learning optimizer*                                                           | [github](https://github.com/lessw2020/Ranger21)                                                                | <https://arxiv.org/abs/2106.13731>                                                          | [cite](https://github.com/lessw2020/Ranger21#referencing-this-work)                                                                 |
| Lamb          | *Large Batch Optimization for Deep Learning*                                                      | [github](https://github.com/cybertronai/pytorch-lamb)                                                          | <https://arxiv.org/abs/1904.00962>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190400962Y/exportcitation)                                                        |
| Shampoo       | *Preconditioned Stochastic Tensor Optimization*                                                   | [github](https://github.com/moskomule/shampoo.pytorch)                                                         | <https://arxiv.org/abs/1802.09568>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv180209568G/exportcitation)                                                        |
| Nero          | *Learning by Turning: Neural Architecture Aware Optimisation*                                     | [github](https://github.com/jxbz/nero)                                                                         | <https://arxiv.org/abs/2102.07227>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210207227L/exportcitation)                                                        |
| Adan          | *Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models*                          | [github](https://github.com/sail-sg/Adan)                                                                      | <https://arxiv.org/abs/2208.06677>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2022arXiv220806677X/exportcitation)                                                        |
| Adai          | *Disentangling the Effects of Adaptive Learning Rate and Momentum*                                | [github](https://github.com/zeke-xie/adaptive-inertia-adai)                                                    | <https://arxiv.org/abs/2006.15815>                                                          | [cite](https://github.com/zeke-xie/adaptive-inertia-adai#citing)                                                                    |
| SAM           | *Sharpness-Aware Minimization*                                                                    | [github](https://github.com/davda54/sam)                                                                       | <https://arxiv.org/abs/2010.01412>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv201001412F/exportcitation)                                                        |
| ASAM          | *Adaptive Sharpness-Aware Minimization*                                                           | [github](https://github.com/davda54/sam)                                                                       | <https://arxiv.org/abs/2102.11600>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210211600K/exportcitation)                                                        |
| GSAM          | *Surrogate Gap Guided Sharpness-Aware Minimization*                                               | [github](https://github.com/juntang-zhuang/GSAM)                                                               | <https://openreview.net/pdf?id=edONMAnhLu->                                                 | [cite](https://github.com/juntang-zhuang/GSAM#citation)                                                                             |
| D-Adaptation  | *Learning-Rate-Free Learning by D-Adaptation*                                                     | [github](https://github.com/facebookresearch/dadaptation)                                                      | <https://arxiv.org/abs/2301.07733>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2023arXiv230107733D/exportcitation)                                                        |
| AdaFactor     | *Adaptive Learning Rates with Sublinear Memory Cost*                                              | [github](https://github.com/DeadAt0m/adafactor-pytorch)                                                        | <https://arxiv.org/abs/1804.04235>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv180404235S/exportcitation)                                                        |
| Apollo        | *An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization*   | [github](https://github.com/XuezheMax/apollo)                                                                  | <https://arxiv.org/abs/2009.13586>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv200913586M/exportcitation)                                                        |
| NovoGrad      | *Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks*      | [github](https://github.com/lonePatient/NovoGrad-pytorch)                                                      | <https://arxiv.org/abs/1905.11286>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190511286G/exportcitation)                                                        |
| Lion          | *Symbolic Discovery of Optimization Algorithms*                                                   | [github](https://github.com/google/automl/tree/master/lion)                                                    | <https://arxiv.org/abs/2302.06675>                                                          | [cite](https://github.com/google/automl/tree/master/lion#citation)                                                                  |
| Ali-G         | *Adaptive Learning Rates for Interpolation with Gradients*                                        | [github](https://github.com/oval-group/ali-g)                                                                  | <https://arxiv.org/abs/1906.05661>                                                          | [cite](https://github.com/oval-group/ali-g#adaptive-learning-rates-for-interpolation-with-gradients)                                |
| SM3           | *Memory-Efficient Adaptive Optimization*                                                          | [github](https://github.com/google-research/google-research/tree/master/sm3)                                   | <https://arxiv.org/abs/1901.11150>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190111150A/exportcitation)                                                        |
| AdaNorm       | *Adaptive Gradient Norm Correction based Optimizer for CNNs*                                      | [github](https://github.com/shivram1987/AdaNorm)                                                               | <https://arxiv.org/abs/2210.06364>                                                          | [cite](https://github.com/shivram1987/AdaNorm/tree/main#citation)                                                                   |
| RotoGrad      | *Gradient Homogenization in Multitask Learning*                                                   | [github](https://github.com/adrianjav/rotograd)                                                                | <https://openreview.net/pdf?id=T8wHz4rnuGL>                                                 | [cite](https://github.com/adrianjav/rotograd#citing)                                                                                |
| A2Grad        | *Optimal Adaptive and Accelerated Stochastic Gradient Descent*                                    | [github](https://github.com/severilov/A2Grad_optimizer)                                                        | <https://arxiv.org/abs/1810.00553>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv181000553D/exportcitation)                                                        |
| AccSGD        | *Accelerating Stochastic Gradient Descent For Least Squares Regression*                           | [github](https://github.com/rahulkidambi/AccSGD)                                                               | <https://arxiv.org/abs/1704.08227>                                                          | [cite](https://github.com/rahulkidambi/AccSGD#citation)                                                                             |
| SGDW          | *Decoupled Weight Decay Regularization*                                                           | [github](https://github.com/loshchil/AdamW-and-SGDW)                                                           | <https://arxiv.org/abs/1711.05101>                                                          | [cite](https://github.com/loshchil/AdamW-and-SGDW#contact)                                                                          |
| ASGD          | *Adaptive Gradient Descent without Descent*                                                       | [github](https://github.com/ymalitsky/adaptive_GD)                                                             | <https://arxiv.org/abs/1910.09529>                                                          | [cite](https://github.com/ymalitsky/adaptive_GD#reference)                                                                          |
| Yogi          | *Adaptive Methods for Nonconvex Optimization*                                                     |                                                                                                                | [NIPS 2018](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization)  | [cite](https://proceedings.neurips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)                   |
| SWATS         | *Improving Generalization Performance by Switching from Adam to SGD*                              |                                                                                                                | <https://arxiv.org/abs/1712.07628>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2017arXiv171207628S/exportcitation)                                                        |
| Fromage       | *On the distance between two neural networks and the stability of learning*                       | [github](https://github.com/jxbz/fromage)                                                                      | <https://arxiv.org/abs/2002.03432>                                                          | [cite](https://github.com/jxbz/fromage#citation)                                                                                    |
| MSVAG         | *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients*                       | [github](https://github.com/lballes/msvag)                                                                     | <https://arxiv.org/abs/1705.07774>                                                          | [cite](https://github.com/lballes/msvag#citation)                                                                                   |
| AdaMod        | *An Adaptive and Momental Bound Method for Stochastic Learning*                                   | [github](https://github.com/lancopku/AdaMod)                                                                   | <https://arxiv.org/abs/1910.12249>                                                          | [cite](https://github.com/lancopku/AdaMod#citation)                                                                                 |
| AggMo         | *Aggregated Momentum: Stability Through Passive Damping*                                          | [github](https://github.com/AtheMathmo/AggMo)                                                                  | <https://arxiv.org/abs/1804.00325>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv180400325L/exportcitation)                                                        |
| QHAdam        | *Quasi-hyperbolic momentum and Adam for deep learning*                                            | [github](https://github.com/facebookresearch/qhoptim)                                                          | <https://arxiv.org/abs/1810.06801>                                                          | [cite](https://github.com/facebookresearch/qhoptim#reference)                                                                       |
| PID           | *A PID Controller Approach for Stochastic Optimization of Deep Networks*                          | [github](https://github.com/tensorboy/PIDOptimizer)                                                            | [CVPR 18](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf)                     | [cite](https://github.com/tensorboy/PIDOptimizer#citation)                                                                          |
| Gravity       | *a Kinematic Approach on Optimization in Deep Learning*                                           | [github](https://github.com/dariush-bahrami/gravity.optimizer)                                                 | <https://arxiv.org/abs/2101.09192>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210109192B/exportcitation)                                                        |
| AdaSmooth     | *An Adaptive Learning Rate Method based on Effective Ratio*                                       |                                                                                                                | <https://arxiv.org/abs/2204.00825v1>                                                        | [cite](https://ui.adsabs.harvard.edu/abs/2022arXiv220400825L/exportcitation)                                                        |
| SRMM          | *Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates* | [github](https://github.com/HanbaekLyu/SRMM)                                                                   | <https://arxiv.org/abs/2201.01652>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2022arXiv220101652L/exportcitation)                                                        |
| AvaGrad       | *Domain-independent Dominance of Adaptive Methods*                                                | [github](https://github.com/lolemacs/avagrad)                                                                  | <https://arxiv.org/abs/1912.01823>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv191201823S/exportcitation)                                                        |
| PCGrad        | *Gradient Surgery for Multi-Task Learning*                                                        | [github](https://github.com/tianheyu927/PCGrad)                                                                | <https://arxiv.org/abs/2001.06782>                                                          | [cite](https://github.com/tianheyu927/PCGrad#reference)                                                                             |
| AMSGrad       | *On the Convergence of Adam and Beyond*                                                           |                                                                                                                | <https://openreview.net/pdf?id=ryQu7f-RZ>                                                   | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190409237R/exportcitation)                                                        |
| Lookahead     | *k steps forward, 1 step back*                                                                    | [github](https://github.com/pytorch/examples/tree/main/imagenet)                                               | <https://arxiv.org/abs/1907.08610>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190708610Z/exportcitation)                                                        |
| PNM           | *Manipulating Stochastic Gradient Noise to Improve Generalization*                                | [github](https://github.com/zeke-xie/Positive-Negative-Momentum)                                               | <https://arxiv.org/abs/2103.17182>                                                          | [cite](https://github.com/zeke-xie/Positive-Negative-Momentum#citing)                                                               |
| GC            | *Gradient Centralization*                                                                         | [github](https://github.com/Yonghongwei/Gradient-Centralization)                                               | <https://arxiv.org/abs/2004.01461>                                                          | [cite](https://github.com/Yonghongwei/Gradient-Centralization#citation)                                                             |
| AGC           | *Adaptive Gradient Clipping*                                                                      | [github](https://github.com/deepmind/deepmind-research/tree/master/nfnets)                                     | <https://arxiv.org/abs/2102.06171>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210206171B/exportcitation)                                                        |
| Stable WD     | *Understanding and Scheduling Weight Decay*                                                       | [github](https://github.com/zeke-xie/stable-weight-decay-regularization)                                       | <https://arxiv.org/abs/2011.11152>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv201111152X/exportcitation)                                                        |
| Softplus T    | *Calibrating the Adaptive Learning Rate to Improve Convergence of ADAM*                           |                                                                                                                | <https://arxiv.org/abs/1908.00700>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190800700T/exportcitation)                                                        |
| Un-tuned w/u  | *On the adequacy of untuned warmup for adaptive optimization*                                     |                                                                                                                | <https://arxiv.org/abs/1910.04209>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv191004209M/exportcitation)                                                        |
| Norm Loss     | *An efficient yet effective regularization method for deep neural networks*                       |                                                                                                                | <https://arxiv.org/abs/2103.06583>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210306583G/exportcitation)                                                        |
| AdaShift      | *Decorrelation and Convergence of Adaptive Learning Rate Methods*                                 | [github](https://github.com/MichaelKonobeev/adashift)                                                          | <https://arxiv.org/abs/1810.00143v4>                                                        | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv181000143Z/exportcitation)                                                        |
| AdaDelta      | *An Adaptive Learning Rate Method*                                                                |                                                                                                                | <https://arxiv.org/abs/1212.5701v1>                                                         | [cite](https://ui.adsabs.harvard.edu/abs/2012arXiv1212.5701Z/exportcitation)                                                        |
| Amos          | *An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale*                 | [github](https://github.com/google-research/jestimator)                                                        | <https://arxiv.org/abs/2210.11693>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2022arXiv221011693T/exportcitation)                                                        |
| SignSGD       | *Compressed Optimisation for Non-Convex Problems*                                                 | [github](https://github.com/jxbz/signSGD)                                                                      | <https://arxiv.org/abs/1802.04434>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2018arXiv180204434B/exportcitation)                                                        |
| Sophia        | *A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*                    | [github](https://github.com/Liuhong99/Sophia)                                                                  | <https://arxiv.org/abs/2305.14342>                                                          | [cite](https://github.com/Liuhong99/Sophia)                                                                                         |
| Prodigy       | *An Expeditiously Adaptive Parameter-Free Learner*                                                | [github](https://github.com/konstmish/prodigy)                                                                 | <https://arxiv.org/abs/2306.06101>                                                          | [cite](https://github.com/konstmish/prodigy#how-to-cite)                                                                            |
| PAdam         | *Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks*    | [github](https://github.com/uclaml/Padam)                                                                      | <https://arxiv.org/abs/1806.06763>                                                          | [cite](https://github.com/uclaml/Padam#citation)                                                                                    |
| LOMO          | *Full Parameter Fine-tuning for Large Language Models with Limited Resources*                     | [github](https://github.com/OpenLMLab/LOMO)                                                                    | <https://arxiv.org/abs/2306.09782>                                                          | [cite](https://github.com/OpenLMLab/LOMO#citation)                                                                                  |
| AdaLOMO       | *Low-memory Optimization with Adaptive Learning Rate*                                             | [github](https://github.com/OpenLMLab/LOMO)                                                                    | <https://arxiv.org/abs/2310.10195>                                                          | [cite](https://github.com/OpenLMLab/LOMO#citation)                                                                                  |
| Tiger         | *A Tight-fisted Optimizer, an optimizer that is extremely budget-conscious*                       | [github](https://github.com/bojone/tiger)                                                                      |                                                                                             | [cite](https://github.com/bojone/tiger/blob/main/README_en.md#citation)                                                             |
| CAME          | *Confidence-guided Adaptive Memory Efficient Optimization*                                        | [github](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/CAME)                            | <https://aclanthology.org/2023.acl-long.243/>                                               | [cite](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/CAME#citation)                                          |
| WSAM          | *Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term*             | [github](https://github.com/intelligent-machine-learning/dlrover/blob/master/atorch/atorch/optimizers/wsam.py) | <https://arxiv.org/abs/2305.15817>                                                          | [cite](https://github.com/intelligent-machine-learning/dlrover)                                                                     |
| Aida          | *A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range*      | [github](https://github.com/guoqiang-zhang-x/Aida-Optimizer)                                                   | <https://arxiv.org/abs/2203.13273>                                                          | [cite](https://github.com/guoqiang-zhang-x/Aida-Optimizer?tab=readme-ov-file#1-brief-description-of-aida)                           |
| GaLore        | *Memory-Efficient LLM Training by Gradient Low-Rank Projection*                                   | [github](https://github.com/jiaweizzhao/GaLore)                                                                | <https://arxiv.org/abs/2403.03507>                                                          | [cite](https://github.com/jiaweizzhao/GaLore/tree/master?tab=readme-ov-file#citation)                                               |
| Adalite       | *Adalite optimizer*                                                                               | [github](https://github.com/VatsaDev/adalite)                                                                  | <https://github.com/VatsaDev/adalite>                                                       | [cite](https://github.com/VatsaDev/adalite)                                                                                         |
| bSAM          | *SAM as an Optimal Relaxation of Bayes*                                                           | [github](https://github.com/team-approx-bayes/bayesian-sam)                                                    | <https://arxiv.org/abs/2210.01620>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2022arXiv221001620M/exportcitation)                                                        |
| Schedule-Free | *Schedule-Free Optimizers*                                                                        | [github](https://github.com/facebookresearch/schedule_free)                                                    | <https://github.com/facebookresearch/schedule_free>                                         | [cite](https://github.com/facebookresearch/schedule_free)                                                                           |
| FAdam         | *Adam is a natural gradient optimizer using diagonal empirical Fisher information*                | [github](https://github.com/lessw2020/fadam_pytorch)                                                           | <https://arxiv.org/abs/2405.12807>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv240512807H/exportcitation)                                                        |
| Grokfast      | *Accelerated Grokking by Amplifying Slow Gradients*                                               | [github](https://github.com/ironjr/grokfast)                                                                   | <https://arxiv.org/abs/2405.20233>                                                          | [cite](https://github.com/ironjr/grokfast?tab=readme-ov-file#citation)                                                              |
| Kate          | *Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad*                     | [github](https://github.com/nazya/KATE)                                                                        | <https://arxiv.org/abs/2403.02648>                                                          | [cite](https://github.com/nazya/KATE?tab=readme-ov-file#remove-that-square-root-a-new-efficient-scale-invariant-version-of-adagrad) |
| StableAdamW   | *Stable and low-precision training for large-scale vision-language models*                        |                                                                                                                | <https://arxiv.org/abs/2304.13013>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2023arXiv230413013W/exportcitation)                                                        |
| AdamMini      | *Use Fewer Learning Rates To Gain More*                                                           | [github](https://github.com/zyushun/Adam-mini)                                                                 | <https://arxiv.org/abs/2406.16793>                                                          | [cite](https://github.com/zyushun/Adam-mini?tab=readme-ov-file#citation)                                                            |
| TRAC          | *Adaptive Parameter-free Optimization*                                                            | [github](https://github.com/ComputationalRobotics/TRAC)                                                        | <https://arxiv.org/abs/2405.16642>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv240516642M/exportcitation)                                                        |
| AdamG         | *Towards Stability of Parameter-free Optimization*                                                |                                                                                                                | <https://arxiv.org/abs/2405.04376>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv240504376P/exportcitation)                                                        |
| AdEMAMix      | *Better, Faster, Older*                                                                           | [github](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch)                                               | <https://arxiv.org/abs/2409.03137>                                                          | [cite](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch?tab=readme-ov-file#reference)                                         |
| SOAP          | *Improving and Stabilizing Shampoo using Adam*                                                    | [github](https://github.com/nikhilvyas/SOAP)                                                                   | <https://arxiv.org/abs/2409.11321>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv240911321V/exportcitation)                                                        |
| ADOPT         | *Modified Adam Can Converge with Any β2 with the Optimal Rate*                                    | [github](https://github.com/iShohei220/adopt)                                                                  | <https://arxiv.org/abs/2411.02853>                                                          | [cite](https://github.com/iShohei220/adopt?tab=readme-ov-file#citation)                                                             |
| FTRL          | *Follow The Regularized Leader*                                                                   |                                                                                                                | <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf> |                                                                                                                                     |
| Cautious      | *Improving Training with One Line of Code*                                                        | [github](https://github.com/kyleliang919/C-Optim)                                                              | <https://arxiv.org/pdf/2411.16085v1>                                                        | [cite](https://github.com/kyleliang919/C-Optim?tab=readme-ov-file#citation)                                                         |
| DeMo          | *Decoupled Momentum Optimization*                                                                 | [github](https://github.com/bloc97/DeMo)                                                                       | <https://arxiv.org/abs/2411.19870>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv241119870P/exportcitation)                                                        |
| MicroAdam     | *Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence*                 | [github](https://github.com/IST-DASLab/MicroAdam)                                                              | <https://arxiv.org/abs/2405.15593>                                                          | [cite](https://github.com/IST-DASLab/MicroAdam?tab=readme-ov-file#citing)                                                           |
| Muon          | *MomentUm Orthogonalized by Newton-schulz*                                                        | [github](https://github.com/KellerJordan/Muon)                                                                 | <https://x.com/kellerjordan0/status/1842300916864844014>                                    | [cite](https://github.com/KellerJordan/Muon)                                                                                        |
| LaProp        | *Separating Momentum and Adaptivity in Adam*                                                      | [github](https://github.com/Z-T-WANG/LaProp-Optimizer)                                                         | <https://arxiv.org/abs/2002.04839>                                                          | [cite](https://github.com/Z-T-WANG/LaProp-Optimizer?tab=readme-ov-file#citation)                                                    |
| APOLLO        | *SGD-like Memory, AdamW-level Performance*                                                        | [github](https://github.com/zhuhanqing/APOLLO)                                                                 | <https://arxiv.org/abs/2412.05270>                                                          | [cite](https://github.com/zhuhanqing/APOLLO?tab=readme-ov-file#-citation)                                                           |
| MARS          | *Unleashing the Power of Variance Reduction for Training Large Models*                            | [github](https://github.com/AGI-Arena/MARS)                                                                    | <https://arxiv.org/abs/2411.10438>                                                          | [cite](https://github.com/AGI-Arena/MARS/tree/main?tab=readme-ov-file#citation)                                                     |
| SGDSaI        | *No More Adam: Learning Rate Scaling at Initialization is All You Need*                           | [github](https://github.com/AnonymousAlethiometer/SGD_SaI)                                                     | <https://arxiv.org/abs/2411.10438>                                                          | [cite](https://github.com/AnonymousAlethiometer/SGD_SaI?tab=readme-ov-file#citation)                                                |
| Grams         | *Gradient Descent with Adaptive Momentum Scaling*                                                 |                                                                                                                | <https://arxiv.org/abs/2412.17107>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv241217107C/exportcitation)                                                        |
| OrthoGrad     | *Grokking at the Edge of Numerical Stability*                                                     | [github](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)                         | <https://arxiv.org/abs/2501.04697>                                                          | [cite](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability?tab=readme-ov-file#citation)                    |
| Adam-ATAN2    | *Scaling Exponents Across Parameterizations and Optimizers*                                       |                                                                                                                | <https://arxiv.org/abs/2407.05872>                                                          | [cite](https://ui.adsabs.harvard.edu/abs/2024arXiv240705872E/exportcitation)                                                        |

## Supported LR Scheduler

You can check the supported learning rate schedulers with below code.

```python
from pytorch_optimizer import get_supported_lr_schedulers

supported_lr_schedulers = get_supported_lr_schedulers()
```

or you can also search them with the filter(s).

```python
from pytorch_optimizer import get_supported_lr_schedulers

get_supported_lr_schedulers('cosine*')
# ['cosine', 'cosine_annealing', 'cosine_annealing_with_warm_restart', 'cosine_annealing_with_warmup']

get_supported_lr_schedulers(['cosine*', '*warm*'])
# ['cosine', 'cosine_annealing', 'cosine_annealing_with_warm_restart', 'cosine_annealing_with_warmup', 'warmup_stable_decay']
```

| LR Scheduler    | Description                                                                     | Official Code                                                                                                                       | Paper                              | Citation                                                                                           |
|-----------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|----------------------------------------------------------------------------------------------------|
| Explore-Exploit | *Wide-minima Density Hypothesis and the Explore-Exploit Learning Rate Schedule* |                                                                                                                                     | <https://arxiv.org/abs/2003.03977> | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv200303977I/exportcitation)                       |
| Chebyshev       | *Acceleration via Fractal Learning Rate Schedules*                              |                                                                                                                                     | <https://arxiv.org/abs/2103.01338> | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210301338A/exportcitation)                       |
| REX             | *Revisiting Budgeted Training with an Improved Schedule*                        | [github](https://github.com/Nerogar/OneTrainer/blob/2c6f34ea0838e5a86774a1cf75093d7e97c70f03/modules/util/lr_scheduler_util.py#L66) | <https://arxiv.org/abs/2107.04197> | [cite](https://ui.adsabs.harvard.edu/abs/2021arXiv210704197C/exportcitation)                       |
| WSD             | *Warmup-Stable-Decay learning rate scheduler*                                   | [github](https://github.com/OpenBMB/MiniCPM)                                                                                        | <https://arxiv.org/abs/2404.06395> | [cite](https://github.com/OpenBMB/MiniCPM?tab=readme-ov-file#%E5%B7%A5%E4%BD%9C%E5%BC%95%E7%94%A8) |

## Supported Loss Function

You can check the supported loss functions with below code.

```python
from pytorch_optimizer import get_supported_loss_functions

supported_loss_functions = get_supported_loss_functions()
```

or you can also search them with the filter(s).

```python
from pytorch_optimizer import get_supported_loss_functions

get_supported_loss_functions('*focal*')
# ['bcefocalloss', 'focalcosineloss', 'focalloss', 'focaltverskyloss']

get_supported_loss_functions(['*focal*', 'bce*'])
# ['bcefocalloss', 'bceloss', 'focalcosineloss', 'focalloss', 'focaltverskyloss']
```

| Loss Functions  | Description                                                                                                             | Official Code                                          | Paper                              | Citation                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------|------------------------------------------------------------------------------|
| Label Smoothing | *Rethinking the Inception Architecture for Computer Vision*                                                             |                                                        | <https://arxiv.org/abs/1512.00567> | [cite](https://ui.adsabs.harvard.edu/abs/2015arXiv151200567S/exportcitation) |
| Focal           | *Focal Loss for Dense Object Detection*                                                                                 |                                                        | <https://arxiv.org/abs/1708.02002> | [cite](https://ui.adsabs.harvard.edu/abs/2017arXiv170802002L/exportcitation) |
| Focal Cosine    | *Data-Efficient Deep Learning Method for Image Classification Using Data Augmentation, Focal Cosine Loss, and Ensemble* |                                                        | <https://arxiv.org/abs/2007.07805> | [cite](https://ui.adsabs.harvard.edu/abs/2020arXiv200707805K/exportcitation) |
| LDAM            | *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss*                                                | [github](https://github.com/kaidic/LDAM-DRW)           | <https://arxiv.org/abs/1906.07413> | [cite](https://github.com/kaidic/LDAM-DRW#reference)                         |
| Jaccard (IOU)   | *IoU Loss for 2D/3D Object Detection*                                                                                   |                                                        | <https://arxiv.org/abs/1908.03851> | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190803851Z/exportcitation) |
| Bi-Tempered     | *The Principle of Unchanged Optimality in Reinforcement Learning Generalization*                                        |                                                        | <https://arxiv.org/abs/1906.03361> | [cite](https://ui.adsabs.harvard.edu/abs/2019arXiv190600336I/exportcitation) |
| Tversky         | *Tversky loss function for image segmentation using 3D fully convolutional deep networks*                               |                                                        | <https://arxiv.org/abs/1706.05721> | [cite](https://ui.adsabs.harvard.edu/abs/2017arXiv170605721S/exportcitation) |
| Lovasz Hinge    | *A tractable surrogate for the optimization of the intersection-over-union measure in neural networks*                  | [github](https://github.com/bermanmaxim/LovaszSoftmax) | <https://arxiv.org/abs/1705.08790> | [cite](https://github.com/bermanmaxim/LovaszSoftmax#citation)                |

## Useful Resources

Several optimization ideas to regularize & stabilize the training. Most of the ideas are applied in `Ranger21` optimizer.

Also, most of the captures are taken from `Ranger21` paper.

|                                                                                 |                                                                       |                                                                                   |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [Adaptive Gradient Clipping](#adaptive-gradient-clipping)                       | [Gradient Centralization](#gradient-centralization)                   | [Softplus Transformation](#softplus-transformation)                               |
| [Gradient Normalization](#gradient-normalization)                               | [Norm Loss](#norm-loss)                                               | [Positive-Negative Momentum](#positive-negative-momentum)                         |
| [Linear learning rate warmup](#linear-learning-rate-warmup)                     | [Stable weight decay](#stable-weight-decay)                           | [Explore-exploit learning rate schedule](#explore-exploit-learning-rate-schedule) |
| [Lookahead](#lookahead)                                                         | [Chebyshev learning rate schedule](#chebyshev-learning-rate-schedule) | [(Adaptive) Sharpness-Aware Minimization](#adaptive-sharpness-aware-minimization) |
| [On the Convergence of Adam and Beyond](#on-the-convergence-of-adam-and-beyond) | [Improved bias-correction in Adam](#improved-bias-correction-in-adam) | [Adaptive Gradient Norm Correction](#adaptive-gradient-norm-correction)           |

### Adaptive Gradient Clipping

This idea originally proposed in `NFNet (Normalized-Free Network)` paper. `AGC (Adaptive Gradient Clipping)` clips gradients based on the `unit-wise ratio of gradient norms to parameter norms`.

* code : [github](https://github.com/deepmind/deepmind-research/tree/master/nfnets)
* paper : [arXiv](https://arxiv.org/abs/2102.06171)

### Gradient Centralization

|                                                                                                               |
|---------------------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/gradient_centralization.png) |

`Gradient Centralization (GC)` operates directly on gradients by centralizing the gradient to have zero mean.

* code : [github](https://github.com/Yonghongwei/Gradient-Centralization)
* paper : [arXiv](https://arxiv.org/abs/2004.01461)

### Softplus Transformation

By running the final variance denom through the softplus function, it lifts extremely tiny values to keep them viable.

* paper : [arXiv](https://arxiv.org/abs/1908.00700)

### Gradient Normalization

### Norm Loss

|                                                                                                 |
|-------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/norm_loss.png) |

* paper : [arXiv](https://arxiv.org/abs/2103.06583)

### Positive-Negative Momentum

|                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/positive_negative_momentum.png) |

* code : [github](https://github.com/zeke-xie/Positive-Negative-Momentum)
* paper : [arXiv](https://arxiv.org/abs/2103.17182)

### Linear learning rate warmup

|                                                                                                        |
|--------------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/linear_lr_warmup.png) |

* paper : [arXiv](https://arxiv.org/abs/1910.04209)

### Stable weight decay

|                                                                                                           |
|-----------------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/stable_weight_decay.png) |

* code : [github](https://github.com/zeke-xie/stable-weight-decay-regularization)
* paper : [arXiv](https://arxiv.org/abs/2011.11152)

### Explore-exploit learning rate schedule

|                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------|
| ![image](https://raw.githubusercontent.com/kozistr/pytorch_optimizer/main/assets/explore_exploit_lr_schedule.png) |

* code : [github](https://github.com/nikhil-iyer-97/wide-minima-density-hypothesis)
* paper : [arXiv](https://arxiv.org/abs/2003.03977)

### Lookahead

`k` steps forward, 1 step back. `Lookahead` consisting of keeping an exponential moving average of the weights that is updated and substituted to the current weights every `k` lookahead steps (5 by default).

### Chebyshev learning rate schedule

Acceleration via Fractal Learning Rate Schedules.

### (Adaptive) Sharpness-Aware Minimization

Sharpness-Aware Minimization (SAM) simultaneously minimizes loss value and loss sharpness.  
In particular, it seeks parameters that lie in neighborhoods having uniformly low loss.

### On the Convergence of Adam and Beyond

Convergence issues can be fixed by endowing such algorithms with 'long-term memory' of past gradients.

### Improved bias-correction in Adam

With the default bias-correction, Adam may actually make larger than requested gradient updates early in training.

### Adaptive Gradient Norm Correction

Correcting the norm of a gradient in each iteration based on the adaptive training history of gradient norm.

### Cautious optimizer

Updates only occur when the proposed update direction aligns with the current gradient.

### Adam-ATAN2

Adam-atan2 is a new numerically stable, scale-invariant version of Adam that eliminates the epsilon hyperparameter.

## Frequently asked questions

[here](docs/qa.md)

## Visualization

[here](docs/visualization.md)

## Citation

Please cite the original authors of optimization algorithms. You can easily find it in the above table! 
If you use this software, please cite it below. Or you can get it from "cite this repository" button.

    @software{Kim_pytorch_optimizer_optimizer_2021,
        author = {Kim, Hyeongchan},
        month = jan,
        title = {{pytorch_optimizer: optimizer & lr scheduler & loss function collections in PyTorch}},
        url = {https://github.com/kozistr/pytorch_optimizer},
        version = {3.1.0},
        year = {2021}
    }

## Maintainer

Hyeongchan Kim / [@kozistr](http://kozistr.tech/about)
