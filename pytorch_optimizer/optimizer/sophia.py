import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Sophia(Optimizer, BaseOptimizer):
    r"""A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.

        Sophia optimizer needs to update the hessian matrix periodically in training code.

        ::python-code

            k: int = 10  # update period
            num_iters: int = 0
            for epoch in epochs:
                for x, y in dataloader:
                    logits = model(x)

                    loss = loss_fn(logits, y)
                    loss.backward()

                    optimizer.step(bs=4096)  # important! need to pass batch size.
                    optimizer.zero_grad()

                    if num_iters % k == 1:
                        sampled_logits = torch.distributions.Categorical(logits=logits).sample()

                        sampled_loss = loss_fn(logits, sampled_logits)
                        sampled_loss.backward()

                        optimizer.update_hessian()
                        optimizer.zero_grad()

                    num_iters += 1

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param rho: float. rho. 0.03 for 125M Sophia-G, 0.05 for 0.05 for 355M Sophia-G.
        Assume, larger model, bigger rho needs.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-4,
        betas: BETAS = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-15,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'rho': rho,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Sophia'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['hessian'] = torch.zeros_like(p)

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                state = self.state[p]
                if 'hessian' in state:
                    state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None, bs: int = 4096) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p,
                    None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                ratio = (exp_avg.abs() / (group['rho'] * bs * state['hessian'] + group['eps'])).clamp_max_(1.0)

                p.addcmul_(exp_avg.sign(), ratio, value=-group['lr'])

        return loss
