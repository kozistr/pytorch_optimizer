__AUTHORS__ = [
    '@lessw2020',
    '@NestorDemeure',
    # with contributions from :
    '@BrianPugh',
    '@Kayuksel',
    '@TheZothen',
]

import collections
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_optimizer.agc import agc
from pytorch_optimizer.chebyshev_schedule import get_chebyshev_schedule
from pytorch_optimizer.gc import centralize_gradient
from pytorch_optimizer.types import (
    BETAS,
    CLOSURE,
    DEFAULT_PARAMETERS,
    LOSS,
    PARAMS,
)
from pytorch_optimizer.utils import normalize_gradient, unit_norm


class Ranger21(Optimizer):
    """
    Reference : https://github.com/lessw2020/Ranger21/blob/main/ranger21/ranger21.py
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
        params: PARAMS,
        lr: float = 1e-3,
        lookahead_active: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        lookahead_load_at_validation: bool = False,
        use_madgrad: bool = False,
        use_adabelief: bool = False,
        softplus: bool = True,
        beta_softplus: int = 50,
        use_gc: bool = True,
        use_gc_norm: bool = True,
        gc_conv_only: bool = False,
        norm_loss_active: bool = True,
        norm_loss_factor: float = 1e-4,
        use_adaptive_gradient_clipping: bool = True,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        momentum_type: str = 'pnm',
        pnm_momentum_factor: float = 1.0,
        momentum: float = 0.9,
        eps: float = 1e-8,
        num_batches_per_epoch: Optional[int] = None,
        num_epochs: Optional[int] = None,
        use_chebyshev_schedule: bool = False,
        use_warmup: bool = True,
        num_warmup_iterations=None,
        warm_down_active: bool = True,
        warm_down_start_pct: float = 0.72,
        warm_down_min_lr: float = 3e-5,
        weight_decay: float = 1e-4,
        decay_type: str = 'stable',
        warmup_type: str = 'linear',
        warmup_pct_default: float = 0.22,
        logging_active: bool = False,
    ):
        """Ranger optimizer (RAdam + Lookahead + Gradient Centralization, combined into one optimizer)
        :param params: PARAMS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate.
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param use_gc: bool. use Gradient Centralization (both convolution & fc layers)
        :param gc_conv_only: bool. use Gradient Centralization (only convolution layer)
        """
        defaults: DEFAULT_PARAMETERS = dict(
            lr=lr,
            momentum=momentum,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.logging = logging_active

        self.use_madgrad = use_madgrad
        self.core_engine: str = self.get_core_engine(self.use_madgrad)

        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_epochs = num_epochs

        self.use_adabelief = use_adabelief
        self.eps = eps
        self.softplus = softplus
        self.beta_softplus = beta_softplus

        # norm loss
        self.norm_loss_active = norm_loss_active
        self.norm_loss_factor = norm_loss_factor

        # lookahead
        self.lookahead_active = lookahead_active
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_alpha = lookahead_blending_alpha
        self.lookahead_step: int = 0
        self.lookahead_validation_load = lookahead_load_at_validation

        # agc
        self.agc_active = use_adaptive_gradient_clipping
        self.agc_clip_val = agc_clipping_value
        self.agc_eps = agc_eps

        # chebyshev schedule
        self.use_chebyshev_schedule = use_chebyshev_schedule
        self.chebyshev_schedule: Optional[np.ndarray] = None
        if self.use_chebyshev_schedule:
            if num_epochs is None:
                raise ValueError(
                    'cannot produce chebyshev without num epochs info being passed in'
                )
            self.chebyshev_schedule = get_chebyshev_schedule(num_epochs)

        self.total_iterations: int = num_epochs * num_batches_per_epoch
        if not self.total_iterations:
            raise ValueError(
                'missing total iterations, '
                'calculated from num epochs and num iterations per epoch param'
            )

        self.starting_lr = lr
        self.current_lr = lr

        # warmup - we'll use default recommended in Ma/Yarats
        # unless user specifies num iterations
        self.use_warmup = use_warmup
        self.warmup_type = warmup_type
        self.warmup_pct_default = warmup_pct_default
        self.warmup_complete: bool = False

        if num_warmup_iterations is None:
            beta_warmup_iterations: int = math.ceil((2 / (1 - betas[1])))
            beta_pct: float = beta_warmup_iterations / self.total_iterations

            # this can be unreasonable for short runs...
            # so let's compare vs warmup pct % of total epochs
            if beta_pct > 0.45:
                warmup_auto_pct = int(
                    self.warmup_pct_default * self.total_iterations
                )
                self.num_warmup_iterations = warmup_auto_pct
            else:
                self.num_warmup_iterations = beta_warmup_iterations
        else:
            self.num_warmup_iterations = num_warmup_iterations

        # warm down
        self.min_lr = warm_down_min_lr
        self.warm_down_active = warm_down_active
        self.warm_down_lr_delta: float = self.starting_lr - self.min_lr

        if self.warm_down_active:
            self.warm_down_start_pct = warm_down_start_pct
            self.start_warm_down = int(
                self.warm_down_start_pct * num_epochs * num_batches_per_epoch
            )
            self.warm_down_total_iterations = (
                self.total_iterations - self.start_warm_down
            )
            self.warmup_curr_pct: float = 0.01

        self.current_epoch = 0
        self.current_iter = 0

        # gradient centralization
        self.use_gc = use_gc
        self.use_gc_norm = use_gc_norm
        self.gc_conv_only = gc_conv_only

        self.epoch_count: int = 0
        self.momentum_pnm: bool = momentum_type == 'pnm'
        self.pnm_momentum = pnm_momentum_factor

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size: int = 0

        self.tracking_lr: List[float] = []
        if self.logging:
            self.tracking_variance_sum: List[float] = []
            self.tracking_variance_normalized = []

    @staticmethod
    def get_core_engine(use_madgrad: bool = False) -> str:
        return 'AdamW' if not use_madgrad else 'madgrad'

    def __setstate__(self, state: Dict):
        super().__setstate__(state)

    def clear_cache(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                if 'lookahead_params' in param_state:
                    la_params = param_state['lookahead_params']
                else:
                    return

                if len(la_params):
                    param_state['lookahead_params'] = torch.zeros_like(p.data)

    def clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def backup_and_load_cache(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['lookahead_params'])

    def warmup_dampening(self, lr: float, step: int) -> float:
        style: str = self.warmup_type
        warmup: int = self.num_warmup_iterations

        if style is None:
            return lr

        if step > warmup:
            if not self.warmup_complete:
                if not self.warmup_curr_pct == 1.0:
                    print(
                        f'Error | lr did not achieve full set point from warmup, currently {self.warmup_curr_pct}'
                    )

                self.warmup_complete = True
                print(
                    f'\n** Ranger21 update | Warmup complete - lr set to {lr}\n'
                )

            return lr

        if style == 'linear':
            self.warmup_curr_pct = min(1.0, (step / warmup))
            new_lr: float = lr * self.warmup_curr_pct
            self.current_lr = new_lr
            return new_lr
        else:
            raise NotImplementedError(
                f'warmup style {style} is not supported yet :('
            )

    def get_warm_down(self, lr: float, iteration: int) -> float:
        if iteration < self.start_warm_down:
            return lr

        if iteration > self.start_warm_down - 1:
            # start iteration from 1, not 0
            warm_down_iteration: int = (iteration + 1) - self.start_warm_down
            if warm_down_iteration < 1:
                warm_down_iteration = 1

            warm_down_pct: float = warm_down_iteration / (
                self.warm_down_total_iterations + 1
            )
            if warm_down_pct > 1.00:
                warm_down_pct = 1.00

            lr_range: float = self.warm_down_lr_delta
            reduction: float = lr_range * warm_down_pct
            new_lr: float = self.starting_lr - reduction
            if new_lr < self.min_lr:
                new_lr = self.min_lr

            self.current_lr = new_lr

            return new_lr

    def track_epochs(self, _: int):
        self.current_iter += 1
        if self.current_iter % self.num_batches_per_epoch == 0:
            self.current_iter = 0
            self.epoch_count += 1
            self.tracking_lr.append(self.current_lr)

            if self.lookahead_active and self.lookahead_validation_load:
                self.backup_and_load_cache()

    def get_chebyshev_lr(self, lr: float, iteration: int) -> float:
        # first confirm we are done with warmup
        if self.use_warmup:
            if iteration < self.num_warmup_iterations + 1:
                return lr

        current_epoch: int = (iteration // self.num_batches_per_epoch) + 1
        self.current_epoch = current_epoch

        index: int = current_epoch - 2
        if index < 0:
            index = 0
        if index > len(self.chebyshev_schedule) - 1:
            index = len(self.chebyshev_schedule) - 1

        chebyshev_value = self.chebyshev_schedule[index]

        if self.cheb_logging[:-1] != chebyshev_value:
            self.cheb_logging.append(chebyshev_value)

        return lr * chebyshev_value

    def get_variance(self):
        return self.tracking_variance_sum

    @staticmethod
    def get_state_values(group, state):
        beta1, beta2 = group['betas']
        mean_avg = state['mean_avg']
        variance_avg = state['variance_avg']
        return beta1, beta2, mean_avg, variance_avg

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None and isinstance(closure, collections.Callable):
            with torch.enable_grad():
                loss = closure()

        param_size: float = 0
        variance_ma_sum: float = 1.0

        # phase 1 - accumulate all of the variance_ma_sum to use in stable weight decay
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                param_size += p.numel()

                if self.agc_active:
                    agc(
                        p, agc_eps=self.agc_eps, agc_clip_val=self.agc_clip_val
                    )

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('sparse matrix not supported atm')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['grad_ma'] = torch.zeros_like(p)
                    state['variance_ma'] = torch.zeros_like(p)

                    if self.lookahead_active:
                        state['lookahead_params'] = torch.zeros_like(p.data)
                        state['lookahead_params'].copy_(p.data)

                    if self.use_adabelief:
                        state['variance_ma_belief'] = torch.zeros_like(p)
                    if self.momentum_pnm:
                        state['neg_grad_ma'] = torch.zeros_like(p)
                        state['max_variance_ma'] = torch.zeros_like(p)

                if self.use_gc:
                    grad = centralize_gradient(
                        grad,
                        gc_conv_only=self.gc_conv_only,
                    )

                if self.use_gc_norm:
                    grad = normalize_gradient(grad)

                # phase 1, variance computations
                state['step'] += 1

                step = state['step']

                beta1, beta2 = group['betas']
                grad_ma = state['grad_ma']

                bias_correction2 = 1 - beta2 ** state['step']

                variance_ma = state['variance_ma']
                if self.use_adabelief:
                    variance_ma_belief = state['variance_ma_belief']

                # update the exp averages
                if self.use_adabelief:
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)
                    grad_residual = grad - grad_ma
                    variance_ma_belief.mul_(beta2).addcmul(
                        grad_residual, grad_residual, value=1 - beta2
                    )

                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                variance_ma_de_biased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_de_biased.sum()

        if not self.param_size:
            self.param_size = param_size

        if not self.param_size:
            raise ValueError('failed to set param size')

        # stable weight decay
        if self.use_madgrad:
            variance_normalized = torch.pow(
                variance_ma_sum / param_size, 1 / 3
            )
        else:
            variance_normalized = math.sqrt(variance_ma_sum / param_size)

        if math.isnan(variance_normalized):
            raise RuntimeError('hit nan for variance_normalized')

        # debugging/logging
        if self.logging:
            self.tracking_variance_sum.append(variance_ma_sum.item())
            self.tracking_variance_normalized.append(variance_normalized)

        # phase 2 - apply weight decay and step
        for group in self.param_groups:
            step = state['step']
            decay = group['weight_decay']
            lr = group['lr']
            momentum = group['momentum']

            if self.use_warmup and not self.warmup_complete:
                lr = self.warmup_dampening(lr, step)

            if self.use_chebyshev_schedule and self.warmup_complete:
                lr = self.get_chebyshev_lr(lr, step)

            # warm-down
            if self.warm_down_active:
                lr = self.get_warm_down(lr, step)
                if 0 > lr:
                    raise ValueError(f'{lr} went negative')

            # MADGRAD outer
            _lambda: float = 0.0
            if self.use_madgrad:
                ck: float = 1.0 - momentum
                _lambda: float = lr * math.pow(step, 0.5)

            # stable decay and / or norm loss
            if decay:
                if not self.use_madgrad:
                    p.data.mul_(1 - decay * lr / variance_normalized)
                else:
                    p.data.mul_(1 - decay * _lambda / variance_normalized)

            if self.norm_loss_active:
                u_norm = unit_norm(p.data)
                correction = (
                    2
                    * self.norm_loss_factor
                    * (1 - torch.div(1, u_norm + self.eps))
                )
                p.mul_(1 - lr * correction)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                inner_grad = p.grad

                if self.use_madgrad:
                    if 'grad_sum_sq' not in state:
                        state['grad_sum_sq'] = torch.zeros_like(
                            p.data
                        ).detach()
                        state['s'] = torch.zeros_like(p.data).detach()
                        if momentum != 0:
                            state['x0'] = torch.clone(p.data).detach()

                    if momentum != 0.0 and grad.is_sparse:
                        raise RuntimeError(
                            'momentum != 0 is not compatible with sparse gradients'
                        )

                    # centralize gradients
                    if self.use_gc:
                        inner_grad = centralize_gradient(
                            inner_grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    grad_sum_sq = state['grad_sum_sq']
                    s = state['s']
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3)
                        if self.softplus:
                            rms = F.softplus(rms, beta=self.beta_softplus)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state['x0']

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(inner_grad, inner_grad, value=_lambda)
                    rms = grad_sum_sq.pow(1 / 3)
                    if self.softplus:
                        rms = F.softplus(rms, beta=self.beta_softplus)

                    s.data.add_(inner_grad, alpha=_lambda)

                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)
                else:
                    grad = p.grad
                    beta1, beta2 = group['betas']
                    grad_ma = state['grad_ma']
                    variance_ma = state['variance_ma']

                    if self.momentum_pnm:
                        max_variance_ma = state['max_variance_ma']

                        if state['step'] % 2 == 1:
                            grad_ma, neg_grad_ma = (
                                state['grad_ma'],
                                state['neg_grad_ma'],
                            )
                        else:
                            grad_ma, neg_grad_ma = (
                                state['neg_grad_ma'],
                                state['grad_ma'],
                            )

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if self.momentum_pnm:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(
                            max_variance_ma, variance_ma, out=variance_ma
                        )
                        # Use the max. for normalizing running avg. of gradient
                        denom = (
                            variance_ma.sqrt() / math.sqrt(bias_correction2)
                        ).add_(group['eps'])

                    if self.use_gc:
                        grad = centralize_gradient(
                            grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    if self.use_gc_norm:
                        grad = normalize_gradient(grad)

                    if not self.use_adabelief:
                        grad_ma.mul_(beta1 ** 2).add_(
                            grad, alpha=1 - beta1 ** 2
                        )

                    noise_norm: float = math.sqrt(
                        (1 + beta2) ** 2 + beta2 ** 2
                    )
                    step_size: float = lr / bias_correction1

                    if self.softplus:
                        denom = F.softplus(denom, beta=self.beta_softplus)

                    pnmomentum = (
                        grad_ma.mul(1 + self.momentum_pnm)
                        .add(neg_grad_ma, alpha=-self.momentum_pnm)
                        .mul(1 / noise_norm)
                    )

                    p.addcdiv_(pnmomentum, denom, value=-step_size)

        if self.lookahead_active:
            self.lookahead_process_step()

        self.track_epochs(step)

        return loss

    def lookahead_process_step(self):
        if not self.lookahead_active:
            return

        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state['lookahead_params'],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    param_state['lookahead_params'].copy_(p.data)
