import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS, HUTCHINSON_G


class SophiaH(Optimizer, BaseOptimizer):
    r"""Second-order Clipped Stochastic Optimization

    Requires `loss.backward(create_graph=True)` in order to calculate hessians

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param p: float. clip effective (applied) gradient (p)
    :param update_period: int. number of steps after which to apply hessian approximation
    :param n_samples: int. times to sample `z` for the approximation of the hessian trace
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(self,
                 params: PARAMETERS,
                 lr: float = 6e-2,
                 betas: BETAS = (0.96, 0.99),
                 weight_decay: float = 0.0,
                 weight_decouple: bool = True,
                 fixed_decay: bool = False,
                 p: float = 1e-2,
                 update_period: int = 10,
                 n_samples: int = 1,
                 hessian_distribution: HUTCHINSON_G = 'gaussian',
                 eps: float = 1e-12):

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(p, "p (gradient clip)")

        self.distribution = hessian_distribution
        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'p': p,
            'eps': eps,
        }
        self.n_samples = n_samples
        self.update_period = update_period
        self._step = 0
        super().__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        self._step = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = torch.zeros_like(p)
                state['hessian_moment'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None, hessian: tuple[torch.Tensor] = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if hessian is not None:
            self.set_hessian(hessian)
        elif self._step % self.update_period == 0:
            self.compute_hutchinson_hessian(self.n_samples, distribution=self.distribution)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                # State initialization
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                    state['hessian_moment'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                beta1, beta2 = group['betas']
                momentum, hessian_moment = state['momentum'], state['hessian_moment']

                momentum.mul_(beta1).add_(p.grad, alpha=1.0-beta1)
                if (self._step % self.update_period == 0 or hessian is not None) and 'hessian' in state:
                    hessian_moment.mul_(beta2).add_(state['hessian'], alpha=1.0-beta2)

                # See https://shreyansh26.github.io/post/2023-05-28_sophia_scalable_second_order_optimizer_llms/#per-coordinate-clipping
                # The official implementation uses a different method to achieve the same thing (might be faster?):
                # https://github.com/Liuhong99/Sophia/blob/bff9df9b584e2084fe037af1ab38f4db31f0acca/sophia.py#L201
                update = torch.clip(momentum/torch.clip(hessian_moment, group['eps']), -group['p'], group['p'])
                p.add_(update, alpha=-group['lr'])

        self._step += 1
        return loss
