import torch
from torch.optim import Optimizer

from pytorch_optimizer.types import (
    BETAS,
    CLOSURE,
    DEFAULT_PARAMETERS,
    LOSS,
    PARAMS,
)


class AdaHessian(Optimizer):
    """
    Reference : https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py
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
        params: PARAMS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        hessian_power: float = 1.0,
        update_each: int = 1,
        n_samples: int = 1,
        average_conv_kernel: bool = False,
        seed: int = 2147483647,
    ):
        """
        :param params: PARAMS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate.
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param hessian_power: float. exponent of the hessian trace
        :param update_each: int. compute the hessian trace approximation only after *this* number of steps
        :param n_samples: int. how many times to sample `z` for the approximation of the hessian trace
        :param average_conv_kernel: bool. average out the hessian traces of convolutional kernels as in the paper.
        :param seed: int.
        """
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        self.hessian_power = hessian_power
        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel
        self.seed = seed

        self.check_valid_parameters()

        # use a separate generator that deterministically generates the same `z`s across all GPUs
        # in case of distributed training
        self.generator: torch.Generator = torch.Generator().manual_seed(
            self.seed
        )

        defaults: DEFAULT_PARAMETERS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super().__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]['hessian_step'] = 0

    def check_valid_parameters(self):
        if 0.0 > self.lr:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if 0.0 > self.eps:
            raise ValueError(f'Invalid eps : {self.eps}')
        if 0.0 > self.weight_decay:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f'Invalid beta_0 : {self.betas[0]}')
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f'Invalid beta_1 : {self.betas[1]}')
        if not 0.0 <= self.hessian_power < 1.0:
            raise ValueError(f'Invalid hessian_power : {self.hessian_power}')

    def get_params(self):
        """Gets all parameters in all param_groups with gradients"""
        return (
            p
            for group in self.param_groups
            for p in group['params']
            if p.requires_grad
        )

    def zero_hessian(self):
        """Zeros out the accumulated hessian traces."""
        for p in self.get_params():
            if (
                not isinstance(p.hess, float)
                and self.state[p]['hessian_step'] % self.update_each == 0
            ):
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter"""
        params = []
        for p in filter(
            lambda param: param.grad is not None, self.get_params()
        ):
            # compute the trace only each `update_each` step
            if self.state[p]['hessian_step'] % self.update_each == 0:
                params.append(p)
            self.state[p]['hessian_step'] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:
            # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(
                self.seed
            )

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [
                torch.randint(
                    0, 2, p.size(), generator=self.generator, device=p.device
                )
                * 2.0
                - 1.0
                for p in params
            ]
            # note that, possible memory leak due to retrain_graph=True
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=i < self.n_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                # approximate the expected values of z * (H@z)
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = (
                        torch.abs(p.hess)
                        .mean(dim=[2, 3], keepdim=True)
                        .expand_as(p.hess)
                        .clone()
                    )

                # Perform correct step-weight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = (
                    state['exp_avg'],
                    state['exp_hessian_diag_sq'],
                )
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    p.hess, p.hess, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq / bias_correction2)
                    .pow_(k / 2)
                    .add_(group['eps'])
                )

                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
