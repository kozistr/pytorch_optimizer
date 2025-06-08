import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


def closest_smaller_divisor_of_n_to_k(n: int, k: int) -> int:
    r"""Get closest smaller divisor of n to k."""
    if n % k == 0:
        return k

    if n <= 1 or k <= 1:
        raise ValueError

    for i in range(k, 0, -1):
        if n % i == 0:
            return i
    return -1


class AdamWSN(BaseOptimizer):
    r"""Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees.

    .. code-block:: python

        sn_params = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
        sn_param_ids = [id(p) for p in sn_params]
        regular_params = [p for p in model.parameters() if id(p) not in sn_param_ids]
        param_groups = [{'params': regular_params, 'sn': False}, {'params': sn_params, 'sn': True}]
        optimizer = AdamWSN(param_groups, lr=args.lr, weight_decay=args.weight_decay, subset_size=args.subset_size)

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param subset_size: int. If you do not know what subset_size to set, a good rule of thumb is to set it as d/2 where
        d is the hidden dimension of your transformer model. For example, the hidden dimension is 4096 for Llama 7B and
        so a good subset_size could be 2048. You can leave the subset_size argument to its default value of -1 to use
        the recommended subset size as stated above.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        subset_size: int = -1,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'subset_size': subset_size,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamWSN'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(grad)

                if group.get('sn'):
                    size: int = grad.numel()

                    if 'subset_size' not in state:
                        state['subset_size'] = closest_smaller_divisor_of_n_to_k(
                            size,
                            (
                                group['subset_size']
                                if group['subset_size'] > 0
                                else int(math.sqrt(size) / abs(int(group['subset_size'])))
                            ),
                        )

                    reshaped_grad = grad.view(size // state['subset_size'], state['subset_size'])
                    second_moment_update = torch.sum(reshaped_grad ** 2, dim=1, keepdim=True)  # fmt: skip
                    state['exp_avg_sq'] = torch.zeros_like(second_moment_update)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] * bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                size = grad.numel()

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if group.get('sn'):
                    reshaped_grad = grad.view(size // state['subset_size'], state['subset_size'])
                    second_moment_update = torch.sum(reshaped_grad**2, dim=1, keepdim=True)
                else:
                    second_moment_update = grad.pow(2)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                if group.get('sn'):
                    numerator = exp_avg.view(size // state['subset_size'], state['subset_size'])
                    norm_grad = (numerator / de_nom).reshape(p.shape)
                    p.add_(norm_grad, alpha=-step_size)
                else:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

        return loss
