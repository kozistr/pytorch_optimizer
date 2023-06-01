import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

# Modified from https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py (MIT David Samuel)


class AdaHessian(Optimizer, BaseOptimizer):
    r"""An Adaptive Second Order Optimizer for Machine Learning

    Requires `loss.backward(create_graph=True)` in order to calculate hessians

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param hessian_power: float. exponent of the hessian trace
    :param update_period: int. number of steps after which to apply hessian approximation
    :param n_samples: int. times to sample `z` for the approximation of the hessian trace
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(self,
                 params: PARAMETERS,
                 lr: float = 1e-1,
                 betas: BETAS = (0.9, 0.999),
                 weight_decay: float = 0.0,
                 weight_decouple: bool = True,
                 fixed_decay: bool = False,
                 hessian_power: float = 1.0,
                 update_period: int = 1,
                 n_samples: int = 1,
                 eps: float = 1e-16):

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_range(hessian_power, "Hessian Power", 0, 1, range_type='(]')

        self.rng_state = None

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'hessian_power': hessian_power,
            'update_period': update_period,
            'n_samples': n_samples,
            'eps': eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            group['hessian_step'] = 0
            for p in group['params']:
                p.hess = None

    def zero_hessian(self):
        for group in self.param_groups:
            if group['hessian_step'] % self.update_each != 0:
                continue

            for p in group['params']:
                if p.hess is not None:
                    p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for group in self.param_groups:
            if group['hessian_step'] % self.update_each == 0:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)

            group['hessian_step'] += 1

        if len(params) == 0:
            return

        generator = torch.Generator(params[0].device)
        if self.rng_state is not None:
            generator.set_state(self.rng_state)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [torch.randint(0, 2, p.size(), generator=generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

        self.rng_state = generator.get_state()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # TODO: check if per-group step is really useful, if not mod here
        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                # State initialization
                state = self.state[p]
                if len(state) <= 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
