# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class MADGRAD(Optimizer):
    """
    Reference 1 : https://github.com/facebookresearch/madgrad
    Reference 2 : https://github.com/lessw2020/Best-Deep-Learning-Optimizers
    Example :
        from pytorch_optimizer import MADGRAD
        ...
        model = YourModel()
        optimizer = MADGRAD(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        """A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic (slightly modified)
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate.
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
            MADGRAD optimizer requires less weight decay than other methods, often as little as zero
            On sparse problems both weight_decay and momentum should be set to 0.
        """
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps

        self.check_valid_parameters()

        defaults: DEFAULTS = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def check_valid_parameters(self):
        if self.lr < 0.0:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if self.weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if not 0.0 < self.momentum <= 1.0:
            raise ValueError(f'Invalid momentum : {self.momentum}')
        if self.eps < 0.0:
            raise ValueError(f'Invalid eps : {self.eps}')

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under optimizer sharding
        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)

        k = self.state['k'].item()

        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr'] + eps
            decay = group['weight_decay']
            momentum = group['momentum']

            ck: float = 1.0 - momentum
            _lambda = lr * math.pow(k + 1, 0.5)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'grad_sum_sq' not in state:
                    state['grad_sum_sq'] = torch.zeros_like(p.data).detach()
                    state['s'] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state['x0'] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError('momentum != 0 is not compatible with sparse gradients')

                grad_sum_sq = state['grad_sum_sq']
                s = state['s']

                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError('weight_decay option is not compatible with sparse gradients')

                    # original implementation
                    # grad.add_(p.data, alpha=decay)

                    # Apply weight decay - L2 / AdamW style
                    p.data.mul_(1 - lr * decay)

                if grad.is_sparse:
                    grad = grad.coalesce()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_values = grad_sum_sq_masked.data.pow(1 / 3).add_(eps)
                    x0_masked_values = p_masked.data.addcdiv(s_masked.data, rms_masked_values, value=1)

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=_lambda)
                    grad_sum_sq_masked.add_(grad_sq, alpha=_lambda)

                    rms_masked_values = grad_sum_sq_masked.data.pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=_lambda)
                    s_masked.data.add_(grad.data, alpha=_lambda)

                    # update masked copy of p
                    p_kp1_masked_values = x0_masked_values.addcdiv(s_masked.data, rms_masked_values, value=-1)

                    # Copy updated masked p to dense p using an add operation
                    p_masked.data.add_(p_kp1_masked_values, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state['x0']

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=_lambda)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    s.data.add_(grad, alpha=_lambda)

                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        p.data.mul_(1 - ck).add_(z, alpha=ck)

        self.state['k'] += 1

        return loss
