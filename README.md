# pytorch_optimizer

Bunch of optimizer implementations in PyTorch with clean-code.

## Usage

## Supported Optimizers

| Optimizer | Description | Official Code | Paper | Note |
| :---: | :---: | :---: | :---: | :---: |
| AdamP | *Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights* | [github](https://github.com/clovaai/AdamP) | [https://arxiv.org/abs/2006.08217](https://arxiv.org/abs/2006.08217) | |
| Adaptive Gradient Clipping (AGC) | *High-Performance Large-Scale Image Recognition Without Normalization* | [github](https://github.com/deepmind/deepmind-research/tree/master/nfnets) | [https://arxiv.org/abs/2102.06171](https://arxiv.org/abs/2102.06171) | |
| Chebyshev LR Schedules | *Acceleration via Fractal Learning Rate Schedules* | [~~github~~]() | [https://arxiv.org/abs/2103.01338v1](https://arxiv.org/abs/2103.01338v1) | |
| Gradient Centralization (GC) | *A New Optimization Technique for Deep Neural Networks* | [github](https://github.com/Yonghongwei/Gradient-Centralization) | [https://arxiv.org/abs/2004.01461](https://arxiv.org/abs/2004.01461) | |
| Lookahead | *k steps forward, 1 step back* | [github](https://github.com/alphadl/lookahead.pytorch) | [https://arxiv.org/abs/1907.08610v2](https://arxiv.org/abs/1907.08610v2) | |
| RAdam | *On the Variance of the Adaptive Learning Rate and Beyond* | [github](https://github.com/LiyuanLucasLiu/RAdam) | [https://arxiv.org/abs/1908.03265](https://arxiv.org/abs/1908.03265) | |
| Ranger | *a synergistic optimizer combining RAdam and LookAhead, and now GC in one optimizer* | [github](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) | | |
| Ranger21 | *integrating the latest deep learning components into a single optimizer* | [github](https://github.com/lessw2020/Ranger21) | | |

## Citations

### AdamP

```
@inproceedings{heo2021adamp,
    title={AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights},
    author={Heo, Byeongho and Chun, Sanghyuk and Oh, Seong Joon and Han, Dongyoon and Yun, Sangdoo and Kim, Gyuwan and Uh, Youngjung and Ha, Jung-Woo},
    year={2021},
    booktitle={International Conference on Learning Representations (ICLR)},
}
```

### Adaptive Gradient Clipping (AGC)

```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:2102.06171},
  year={2021}
}
```

### Chebyshev LR Schedules

```
@article{agarwal2021acceleration,
  title={Acceleration via Fractal Learning Rate Schedules},
  author={Agarwal, Naman and Goel, Surbhi and Zhang, Cyril},
  journal={arXiv preprint arXiv:2103.01338},
  year={2021}
}
```

### Gradient Centralization (GC)

```
@inproceedings{yong2020gradient,
  title={Gradient centralization: A new optimization technique for deep neural networks},
  author={Yong, Hongwei and Huang, Jianqiang and Hua, Xiansheng and Zhang, Lei},
  booktitle={European Conference on Computer Vision},
  pages={635--652},
  year={2020},
  organization={Springer}
}
```

### Lookahead

```
@article{zhang2019lookahead,
  title={Lookahead optimizer: k steps forward, 1 step back},
  author={Zhang, Michael R and Lucas, James and Hinton, Geoffrey and Ba, Jimmy},
  journal={arXiv preprint arXiv:1907.08610},
  year={2019}
}
```

### RAdam

```
@inproceedings{liu2019radam,
 author = {Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
 booktitle = {Proceedings of the Eighth International Conference on Learning Representations (ICLR 2020)},
 month = {April},
 title = {On the Variance of the Adaptive Learning Rate and Beyond},
 year = {2020}
}
```

## Author

Hyeongchan Kim / [@kozistr](http://kozistr.tech/about)
