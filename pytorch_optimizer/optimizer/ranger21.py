import math
from typing import Optional

import torch
from torch.nn.functional import softplus

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm


class Ranger21(BaseOptimizer):
    """Integrating the latest deep learning components into a single optimizer.

    Here's the components
        * uses the AdamW optimizer as its core (or, optionally, MadGrad)
        * Adaptive gradient clipping
        * Gradient centralization
        * Positive-Negative momentum
        * Norm loss
        * Stable weight decay
        * Linear learning rate warm-up
        * Explore-exploit learning rate schedule
        * Lookahead
        * Softplus transformation
        * Gradient Normalization
        * Corrects the denominator (AdamD).

    Args:
        params (Parameters): iterable of parameters to optimize or dicts defining parameter groups.
        num_iterations (int): number of the total training steps. Ranger21 optimizer schedules the learning rate
            with its own recipes.
        lr (float): learning rate.
        beta0 (float): Manages the amplitude of the noise introduced by positive negative momentum
            while 0.9 is a recommended default value, you can use -0.5 to minimize the noise.
        betas (Betas): coefficients used for computing running averages of gradient and the squared hessian trace.
        use_softplus (bool): use softplus to smooth.
        beta_softplus (float): beta.
        disable_lr_scheduler (bool): whether to disable learning rate schedule.
        num_warm_up_iterations (Optional[int]): number of warm-up iterations. Ranger21 performs linear learning rate
            warmup.
        num_warm_down_iterations (Optional[int]): number of warm-down iterations. Ranger21 performs Explore-exploit
            learning rate scheduling.
        agc_clipping_value (float):
        agc_eps (float): eps for AGC
        centralize_gradients (bool): use GC both convolution & fc layers.
        normalize_gradients (bool): use gradient normalization.
        lookahead_merge_time (int): merge time.
        lookahead_blending_alpha (float): blending alpha.
        weight_decay (float): weight decay (L2 penalty).
        weight_decouple (bool): the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): fix weight decay.
        norm_loss_factor (float): norm loss factor.
        eps (float): term added to the denominator to improve numerical stability.
        maximize (bool): maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(  # pylint: disable=R0913
        self,
        params: Parameters,
        num_iterations: int,
        lr: float = 1e-3,
        beta0: float = 0.9,
        betas: Betas = (0.9, 0.999),
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        disable_lr_scheduler: bool = False,
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
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        norm_loss_factor: float = 1e-4,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(warm_down_min_lr)
        self.validate_betas(betas)
        self.validate_range(beta0, 'beta0', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(agc_clipping_value, 'agc_clipping_value')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(agc_eps, 'agc_eps')

        self.min_lr = warm_down_min_lr
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.disable_lr_scheduler = disable_lr_scheduler
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.norm_loss_factor = norm_loss_factor
        self.maximize = maximize

        self.lookahead_step: int = 0
        self.starting_lr: float = lr
        self.current_lr: float = lr

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

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

    def __str__(self) -> str:
        return 'Ranger21'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
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
                state['grad_ma'] = torch.zeros_like(p)
                state['variance_ma'] = torch.zeros_like(p)
                state['lookahead_params'] = p.clone()
                state['neg_grad_ma'] = torch.zeros_like(p)
                state['max_variance_ma'] = torch.zeros_like(p)

    @staticmethod
    def build_warm_up_iterations(total_iterations: int, beta2: float, warm_up_pct: float = 0.22) -> int:
        warm_up_iterations: int = math.ceil(2.0 / (1.0 - beta2))  # default un-tuned linear warmup
        beta_pct: float = warm_up_iterations / total_iterations
        return int(warm_up_pct * total_iterations) if beta_pct > 0.45 else warm_up_iterations

    @staticmethod
    def build_warm_down_iterations(total_iterations: int, warm_down_pct: float = 0.72) -> int:
        start_warm_down: int = int(warm_down_pct * total_iterations)
        return total_iterations - start_warm_down

    def warm_up_dampening(self, lr: float, step: int) -> float:
        if step > self.num_warm_up_iterations:
            return lr

        warm_up_current_pct: float = min(1.0, (step / self.num_warm_up_iterations))

        self.current_lr = lr * warm_up_current_pct

        return self.current_lr

    def warm_down(self, lr: float, iteration: int) -> float:
        if iteration < self.start_warm_down:
            return lr

        # start iteration from 1, not 0
        warm_down_iteration: int = max((iteration + 1) - self.start_warm_down, 1)
        warm_down_pct: float = min(warm_down_iteration / (self.num_warm_down_iterations + 1), 1.0)

        self.current_lr = max(self.starting_lr - self.warm_down_lr_delta * warm_down_pct, self.min_lr)

        return self.current_lr

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        variance_ma_sum: float = 1.0

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction2: float = self.debias(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                param_size += p.numel()

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad.copy_(agc(p, grad, self.agc_eps, self.agc_clipping_value))

                centralize_gradient(grad, gc_conv_only=False)
                normalize_gradient(grad)

                variance_ma = state['variance_ma']
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                variance_ma_sum += (variance_ma / bias_correction2).sum()

        if param_size == 0:
            raise ZeroParameterSizeError()

        variance_normalized = math.sqrt(variance_ma_sum / param_size)

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            noise_norm: float = math.sqrt((1.0 + beta2) ** 2 + beta2 ** 2)  # fmt: skip

            if self.disable_lr_scheduler:
                lr: float = group['lr']
            else:
                lr: float = self.warm_up_dampening(group['lr'], group['step'])
                lr = self.warm_down(lr, group['step'])

            step_size: float = self.apply_adam_debias(group.get('adam_debias', False), lr, bias_correction1)

            for p in group['params']:
                if p.grad is None:
                    continue

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=1.0 / variance_normalized,
                )

                correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p).add_(group['eps']))
                p.mul_(1.0 - lr * correction)

                state = self.state[p]
                if group['step'] % 2 == 1:
                    grad_ma, neg_grad_ma = state['grad_ma'], state['neg_grad_ma']
                else:
                    grad_ma, neg_grad_ma = state['neg_grad_ma'], state['grad_ma']

                variance_ma = state['variance_ma']
                torch.max(state['max_variance_ma'], variance_ma, out=variance_ma)

                de_nom = (variance_ma.sqrt() / bias_correction2_sq).add_(group['eps'])

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus)

                grad = p.grad
                centralize_gradient(grad, gc_conv_only=False)
                normalize_gradient(grad)

                grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)  # fmt: skip

                pn_momentum = grad_ma.mul(2.0).add_(neg_grad_ma, alpha=-1.0).mul_(1.0 / noise_norm)
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

                    state = self.state[p]

                    p.mul_(self.lookahead_blending_alpha).add_(
                        state['lookahead_params'],
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )
                    state['lookahead_params'].copy_(p)
