from typing import Iterable, List

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaHessian(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/davda54/ada-hessian
    Example :
        from pytorch_optimizer import AdaHessian
        ...
        model = YourModel()
        optimizer = AdaHessian(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward(create_graph=True)  # this is the important line!
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        hessian_power: float = 1.0,
        update_each: int = 1,
        num_samples: int = 1,
        average_conv_kernel: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
        seed: int = 1337,
    ):
        """AdaHessian
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param hessian_power: float. exponent of the hessian trace
        :param update_each: int. compute the hessian trace approximation only after *this* number of steps
        :param num_samples: int. how many times to sample `z` for the approximation of the hessian trace
        :param average_conv_kernel: bool. average out the hessian traces of convolutional kernels as in the paper
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param eps: float. term added to the denominator to improve numerical stability
        :param seed: int.
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.hessian_power = hessian_power
        self.update_each = update_each
        self.num_samples = num_samples
        self.average_conv_kernel = average_conv_kernel
        self.eps = eps
        self.seed = seed

        self.validate_parameters()

        # use a separate generator that deterministically generates
        # the same `z`s across all GPUs in case of distributed training
        self.generator: torch.Generator = torch.Generator().manual_seed(self.seed)

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
            adamd_debias_term=adamd_debias_term,
        )
        super().__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]['hessian_step'] = 0

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_hessian_power(self.hessian_power)
        self.validate_epsilon(self.eps)

    def get_params(self) -> Iterable:
        """Gets all parameters in all param_groups with gradients"""
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """Zeros out the accumulated hessian traces."""
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]['hessian_step'] % self.update_each == 0:
                p.hess.zero_()

    def set_hessian(self):
        """Computes the Hutchinson approximation of the hessian trace
        and accumulates it for each trainable parameter
        """
        params = []
        for p in self.get_params():
            if p.grad is None:
                continue

            # compute the trace only each `update_each` step
            if self.state[p]['hessian_step'] % self.update_each == 0:
                params.append(p)

            self.state[p]['hessian_step'] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:
            # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)

        grads: List[torch.Tensor] = [p.grad for p in params]

        for i in range(self.num_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [2.0 * torch.randint(0, 2, p.size()).float().requires_grad_(True) - 1.0 for p in params]

            # note that, possible memory leak due to retrain_graph=True
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=i < self.num_samples - 1,
            )

            for h_z, z, p in zip(h_zs, zs, params):
                # approximate the expected values of z * (H@z)
                p.hess += h_z * z / self.num_samples

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_hessian_diag_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=(2, 3), keepdim=True).expand_as(p.hess).clone()

                # Perform correct step-weight decay as in AdamW
                p.mul_(1.0 - group['lr'] * group['weight_decay'])

                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                state['step'] += 1
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                de_nom = (exp_hessian_diag_sq / bias_correction2).pow_(group['hessian_power'] / 2.0).add_(group['eps'])

                step_size = group['lr']
                if not group['adamd_debias_term']:
                    step_size /= bias_correction1

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
