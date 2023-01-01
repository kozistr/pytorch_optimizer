import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.base_optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Adai(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/zeke-xie/adaptive-inertia-adai
    Example :
        from pytorch_optimizer import Adai
        ...
        model = YourModel()
        optimizer = Adai(model.parameters())
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
        betas: BETAS = (0.1, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        eps: float = 1e-3,
    ):
        """Adai
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.param_size: int = self.get_parameter_size()

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['beta1_prod'] = torch.ones_like(p)

    def get_parameter_size(self) -> int:
        param_size: int = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_size += p.numel()

        return param_size

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        exp_avg_sq_hat_sum: float = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adai does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['beta1_prod'] = torch.ones_like(p)

                state['step'] += 1

                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bias_correction2 = 1.0 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    if self.weight_decouple:
                        p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p, alpha=group['weight_decay'])

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2

        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / self.param_size

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adai does not support sparse gradients')

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_prod = state['beta1_prod']
                beta0, beta2 = group['betas']

                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (1.0 - (exp_avg_sq_hat / exp_avg_sq_hat_mean).mul(beta0)).clamp(0.0, 1.0 - group['eps'])

                beta1_prod.mul_(beta1)
                bias_correction1 = 1.0 - beta1_prod

                exp_avg.mul_(beta1).addcmul_(1.0 - beta1, grad)
                exp_avg_hat = exp_avg / bias_correction1

                p.add_(exp_avg_hat, alpha=-group['lr'])

        return loss
