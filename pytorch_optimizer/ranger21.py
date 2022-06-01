import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_optimizer.agc import agc
from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.gc import centralize_gradient
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.utils import normalize_gradient, unit_norm

__AUTHORS__ = [
    '@lessw2020',
    '@NestorDemeure',
    # with contributions from :
    '@BrianPugh',
    '@Kayuksel',
    '@TheZothen',
]


class Ranger21(Optimizer, BaseOptimizer):
    """
    Reference 1 : https://github.com/lessw2020/Ranger21
    Reference 2 : https://github.com/nestordemeure/flaxOptimizers/blob/main/flaxOptimizers/ranger21.py
    Example :
        from pytorch_optimizer import Ranger21
        ...
        model = YourModel()
        optimizer = Ranger21(model.parameters())
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
        num_iterations: int,
        lr: float = 1e-3,
        beta0: float = 0.9,
        betas: BETAS = (0.9, 0.999),
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        num_warm_up_iterations: Optional[int] = None,
        num_warm_down_iterations: Optional[int] = None,
        warm_down_min_lr: float = 3e-5,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
        centralize_gradients: bool = True,
        normalize_gradients: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        weight_decay: float = 1e-4,
        norm_loss_factor: float = 1e-4,
        eps: float = 1e-8,
    ):
        """Ranger21 optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param beta0: float. Manages the amplitude of the noise introduced by positive negative momentum
            While 0.9 is a recommended default value, you can use -0.5 to minimize the noise
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param use_softplus: bool. use softplus to smooth
        :param beta_softplus: float. beta
        :param agc_clipping_value: float.
        :param agc_eps: float. eps for AGC
        :param centralize_gradients: bool. use GC both convolution & fc layers
        :param normalize_gradients: bool. use gradient normalization
        :param lookahead_merge_time: int. merge time
        :param lookahead_blending_alpha: float. blending alpha
        :param weight_decay: float. weight decay (L2 penalty)
        :param norm_loss_factor: float. norm loss factor
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.beta0 = beta0
        self.betas = betas
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.weight_decay = weight_decay
        self.norm_loss_factor = norm_loss_factor
        self.eps = eps

        self.validate_parameters()

        # lookahead
        self.lookahead_step: int = 0

        # learning rate
        self.starting_lr = lr
        self.current_lr = lr
        self.min_lr = warm_down_min_lr

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # warmup iterations
        self.num_warm_up_iterations: int = (
            self.build_warm_up_iterations(num_iterations, betas[1])
            if num_warm_up_iterations is None
            else num_warm_up_iterations
        )
        self.num_warm_down_iterations: int = (
            self.build_warm_down_iterations(num_iterations)
            if num_warm_down_iterations is None
            else num_warm_down_iterations
        )
        self.start_warm_down: int = num_iterations - self.num_warm_down_iterations
        self.warm_down_lr_delta: float = self.starting_lr - self.min_lr

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_beta0(self.beta0)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['grad_ma'] = torch.zeros_like(p)
                state['variance_ma'] = torch.zeros_like(p)
                state['lookahead_params'] = torch.empty_like(p)
                state['lookahead_params'].copy_(p)
                state['neg_grad_ma'] = torch.zeros_like(p)
                state['max_variance_ma'] = torch.zeros_like(p)

    @staticmethod
    def build_warm_up_iterations(total_iterations: int, beta2: float, warm_up_pct: float = 0.22) -> int:
        warm_up_iterations: int = math.ceil(2.0 / (1.0 - beta2))  # default un-tuned linear warmup
        beta_pct: float = warm_up_iterations / total_iterations
        if beta_pct > 0.45:
            return int(warm_up_pct * total_iterations)
        return warm_up_iterations

    @staticmethod
    def build_warm_down_iterations(total_iterations: int, warm_down_pct: float = 0.72) -> int:
        start_warm_down: int = int(warm_down_pct * total_iterations)
        return total_iterations - start_warm_down

    def warm_up_dampening(self, lr: float, step: int) -> float:
        if step > self.num_warm_up_iterations:
            return lr

        warm_up_current_pct: float = min(1.0, (step / self.num_warm_up_iterations))

        new_lr: float = lr * warm_up_current_pct
        self.current_lr = new_lr

        return new_lr

    def get_warm_down(self, lr: float, iteration: int) -> float:
        if iteration < self.start_warm_down:
            return lr

        # start iteration from 1, not 0
        warm_down_iteration: int = (iteration + 1) - self.start_warm_down
        warm_down_iteration = max(warm_down_iteration, 1)

        warm_down_pct: float = warm_down_iteration / (self.num_warm_down_iterations + 1)
        warm_down_pct = min(warm_down_pct, 1.0)

        new_lr: float = self.starting_lr - self.warm_down_lr_delta * warm_down_pct
        new_lr = max(new_lr, self.min_lr)
        self.current_lr = new_lr

        return new_lr

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        variance_ma_sum: float = 1.0

        # Phase 1 - Accumulate all the variance_ma_sum to use in stable weight decay
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Ranger21 does not support sparse gradients')

                param_size += p.numel()

                # Apply Adaptive Gradient Clipping (AGC)
                agc(p, agc_eps=self.agc_eps, agc_clip_val=self.agc_clipping_value)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_ma'] = torch.zeros_like(p)
                    state['variance_ma'] = torch.zeros_like(p)
                    state['lookahead_params'] = torch.empty_like(p)
                    state['lookahead_params'].copy_(p)
                    state['neg_grad_ma'] = torch.zeros_like(p)
                    state['max_variance_ma'] = torch.zeros_like(p)

                # Apply GC & GradNorm
                # TODO : Gradient Clipping (Norm)
                grad = centralize_gradient(grad, gc_conv_only=False)
                grad = normalize_gradient(grad)

                state['step'] += 1

                beta1, beta2 = group['betas']

                bias_correction2 = 1.0 - beta2 ** state['step']

                # second moment estimation
                # using positive-negative momentum and bias correction
                variance_ma = state['variance_ma']
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                variance_ma_sum += (variance_ma / bias_correction2).sum()

        # stable weight decay
        variance_normalized = math.sqrt(variance_ma_sum / param_size)
        if math.isnan(variance_normalized):
            raise RuntimeError('hit nan for variance_normalized')

        # Phase 2 - Apply weight decay and step
        for group in self.param_groups:
            p = group['params'][0]
            if p.grad is None:
                continue

            lr = group['lr']
            step = self.state[group['params'][0]]['step']

            # warm up
            lr = self.warm_up_dampening(lr, step)

            # warm down
            lr = self.get_warm_down(lr, step)
            if lr < 0.0:
                raise ValueError(f'{lr} went negative')

            # stable decay
            decay = group['weight_decay']
            if decay:
                p.mul_(1.0 - decay * lr / variance_normalized)

            # norm loss
            u_norm = unit_norm(p)
            correction = 2.0 * self.norm_loss_factor * (1.0 - torch.div(1, u_norm + self.eps))
            p.mul_(1.0 - lr * correction)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if state['step'] % 2 == 1:
                    grad_ma, neg_grad_ma = state['grad_ma'], state['neg_grad_ma']
                else:
                    grad_ma, neg_grad_ma = state['neg_grad_ma'], state['grad_ma']

                beta1, beta2 = group['betas']
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                variance_ma = state['variance_ma']
                max_variance_ma = state['max_variance_ma']

                torch.max(max_variance_ma, variance_ma, out=variance_ma)
                de_nom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                grad = p.grad
                grad = centralize_gradient(grad, gc_conv_only=False)
                grad = normalize_gradient(grad)

                grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)

                noise_norm: float = math.sqrt((1.0 + beta2) ** 2 + beta2 ** 2)

                step_size: float = lr / bias_correction1

                if self.use_softplus:
                    de_nom = F.softplus(de_nom, beta=self.beta_softplus)

                pn_momentum = grad_ma.mul(1.0 + 1.0).add(neg_grad_ma, alpha=-1.0).mul(1.0 / noise_norm)
                p.addcdiv_(pn_momentum, de_nom, value=-step_size)

        self.lookahead_process_step()

        return loss

    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.mul_(self.lookahead_blending_alpha).add_(
                        param_state['lookahead_params'],
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )
                    param_state['lookahead_params'].copy_(p)
