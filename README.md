# pytorch_optimizer

Bunch of optimizer implementations in PyTorch with clean-code.

## Usage

## Supported Optimizers

| Optimizer | Description | Official Code | Paper | Note |
| :---: | :---: | :---: | :---: | :---: |
| Gradient Centralization | *A New Optimization Technique for Deep Neural Networks* | [github](https://github.com/Yonghongwei/Gradient-Centralization) | [https://arxiv.org/abs/2004.01461](https://arxiv.org/abs/2004.01461) | |
| Lookahead | *k steps forward, 1 step back* | [github](https://github.com/alphadl/lookahead.pytorch) | [https://arxiv.org/abs/1907.08610v2](https://arxiv.org/abs/1907.08610v2) | |
| RAdam | *On the Variance of the Adaptive Learning Rate and Beyond* | [github](https://github.com/LiyuanLucasLiu/RAdam) | [https://arxiv.org/abs/1908.03265](https://arxiv.org/abs/1908.03265) | |
| Ranger | *a synergistic optimizer combining RAdam and LookAhead, and now GC in one optimizer* | [github](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) | | |
| Ranger21 | *integrating the latest deep learning components into a single optimizer* | [github](https://github.com/lessw2020/Ranger21) | | |

## Citations

### Gradient Centralization

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
