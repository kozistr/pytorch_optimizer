from contextlib import ExitStack
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce, get_world_size, is_initialized
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, OPTIMIZER, PARAMETERS
from pytorch_optimizer.optimizer.utils import disable_running_stats, enable_running_stats


class SAM(BaseOptimizer):
    r"""Sharpness-Aware Minimization for Efficiently Improving Generalization.

    Example:
    -------
        Here's an example::

            model = YourModel()
            base_optimizer = Ranger21
            optimizer = SAM(model.parameters(), base_optimizer)

            for input, output in data:
                # first forward-backward pass

                loss = loss_function(output, model(input))
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                # make sure to do a full forward pass
                loss_function(output, model(input)).backward()
                optimizer.second_step(zero_grad=True)

        Alternative example with a single closure-based step function::

            model = YourModel()
            base_optimizer = Ranger21
            optimizer = SAM(model.parameters(), base_optimizer)

            def closure():
                loss = loss_function(output, model(input))
                loss.backward()
                return loss

            for input, output in data:
                loss = loss_function(output, model(input))
                loss.backward()
                optimizer.step(closure)
                optimizer.zero_grad()

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param base_optimizer: Optimizer. base optimizer.
    :param rho: float. size of the neighborhood for computing the max loss.
    :param adaptive: bool. element-wise Adaptive SAM.
    :param perturb_eps: float. eps for perturbation.
    :param kwargs: Dict. parameters for optimizer.
    """

    def __init__(
        self,
        params: PARAMETERS,
        base_optimizer: OPTIMIZER,
        rho: float = 0.05,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.perturb_eps = perturb_eps

        defaults: DEFAULTS = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'SAM'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + self.perturb_eps)

            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data = self.state[p]['old_p']

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None):
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()

    def grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GSAM(BaseOptimizer):  # pragma: no cover
    r"""Surrogate Gap Guided Sharpness-Aware Minimization.

    Example:
    -------
        Here's an example::

            model = YourModel()
            base_optimizer = AdamP(model.parameters())
            lr_scheduler = LinearScheduler(base_optimizer, t_max=num_total_steps)
            rho_scheduler = ProportionScheduler(lr_scheduler, max_lr=max_lr)
            optimizer = GSAM(model.parameters(), base_optimizer, model, rho_scheduler)

            def loss_fn(predictions, targets):
                return F.cross_entropy(predictions, targets)

            for inputs, targets in data:
                optimizer.set_closure(loss_fn, inputs, targets)
                predictions, loss = optimizer.step()
                lr_scheduler.step()
                optimizer.update_rho_t()

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param base_optimizer: Optimizer. base optimizer.
    :param model: nn.Module. model.
    :param alpha: float. rho alpha.
    :param rho_scheduler: rho scheduler.
    :param adaptive: bool. element-wise Adaptive SAM.
    :param perturb_eps: float. epsilon for perturbation.
    :param kwargs: Dict. parameters for optimizer.
    """

    def __init__(
        self,
        params: PARAMETERS,
        base_optimizer: OPTIMIZER,
        model: nn.Module,
        rho_scheduler,
        alpha: float = 0.4,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_range(alpha, 'alpha', 0.0, 1.0)

        self.model = model
        self.rho_scheduler = rho_scheduler
        self.alpha = alpha
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps

        self.rho_t: float = 0.0
        self.forward_backward_func: Optional[Callable] = None

        if hasattr(ReduceOp, 'AVG'):
            self.grad_reduce = ReduceOp.AVG
            self.manual_average: bool = False
        else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
            self.grad_reduce = ReduceOp.SUM
            self.manual_average: bool = True

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        defaults: DEFAULTS = {'adaptive': adaptive}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        self.update_rho_t()

    def __str__(self) -> str:
        return 'GSAM'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def update_rho_t(self) -> float:
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho: float):
        grad_norm = self.grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['old_g'] = p.grad.clone()

                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def un_perturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha: float = 0.0):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                inner_prod += torch.sum(self.state[p]['old_g'] * p.grad)

        new_grad_norm = self.grad_norm(by=None)
        old_grad_norm = self.grad_norm(by='old_g')

        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def sync_grad(self):
        if is_initialized():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    all_reduce(p.grad, op=self.grad_reduce)
                    if self.manual_average:
                        p.grad.div_(float(get_world_size()))

    @torch.no_grad()
    def grad_norm(self, by: Optional[str] = None, weight_adaptive: bool = False) -> torch.Tensor:
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if weight_adaptive else 1.0) * (p.grad if not by else self.state[p][by])).norm(p=2)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def maybe_no_sync(self):
        return self.model.no_sync() if is_initialized() else ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, **kwargs):
        r"""Set closure.

            Create `self.forward_backward_func`, which is a function such that `self.forward_backward_func()`
            automatically performs forward and backward passes. This function does not take any arguments,
            and the inputs and targets data should be pre-set in the definition of partial-function.

        :param loss_fn: nn.Module. loss function.
        :param inputs: torch.Tensor. inputs.
        :param targets: torch.Tensor. targets.
        """

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)

            loss.backward()

            return outputs, loss.detach()

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> Tuple[torch.Tensor, float]:
        get_grad = closure if closure else self.forward_backward_func

        with self.maybe_no_sync():
            outputs, loss = get_grad()

            self.perturb_weights(rho=self.rho_t)

            disable_running_stats(self.model)

            get_grad()

            self.gradient_decompose(self.alpha)

            self.un_perturb()

        self.sync_grad()

        self.base_optimizer.step()

        enable_running_stats(self.model)

        return outputs, loss

    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class WSAM(BaseOptimizer):
    r"""Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term.

    :param model: Union[torch.nn.Module, torch.nn.DataParallel]. the model instance. DDP model is recommended to make
        `model.no_sync` to work.
    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param base_optimizer: Optimizer. base optimizer.
    :param rho: float. size of the neighborhood for computing the max loss.
    :param gamma: float. weighted factor gamma / (1 - gamma) of the sharpness term. 0.8 ~ 0.95 is the optimal.
    :param adaptive: bool. element-wise adaptive SAM.
    :param decouple: bool. whether to perform a decoupled sharpness regularization.
    :param max_norm: Optional[float]. max norm of the gradients.
    :param eps: float. term added to the denominator of WSAM to improve numerical stability.
    :param kwargs: Dict. parameters for optimizer.
    """

    def __init__(
        self,
        model: Union[nn.Module, DistributedDataParallel],
        params: PARAMETERS,
        base_optimizer: OPTIMIZER,
        rho: float = 0.05,
        gamma: float = 0.9,
        adaptive: bool = False,
        decouple: bool = True,
        max_norm: Optional[float] = None,
        eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')

        self.model = model
        self.decouple = decouple
        self.max_norm = max_norm

        alpha: float = gamma / (1.0 - gamma)

        defaults: DEFAULTS = {'rho': rho, 'alpha': alpha, 'adaptive': adaptive, 'sam_eps': eps}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'WSAM'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + group['sam_eps'])

            for p in group['params']:
                if p.grad is None:
                    continue

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]['e_w'] = e_w

                if is_initialized():  # pragma: no cover
                    all_reduce(p.grad, op=ReduceOp.AVG)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['grad'] = p.grad.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if is_initialized():  # pragma: no cover
                    all_reduce(p.grad, ReduceOp.AVG)

                p.add_(self.state[p]['e_w'], alpha=-1.0)

        if self.max_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_norm)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if not self.decouple:
                    p.grad.mul_(group['alpha']).add_(self.state[p]['grad'], alpha=1.0 - group['alpha'])
                else:
                    self.state[p]['sharpness'] = p.grad.clone() - self.state[p]['grad']
                    p.grad.mul_(0.0).add_(self.state[p]['grad'], alpha=1.0)

        self.base_optimizer.step()

        if self.decouple:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    p.add_(self.state[p]['sharpness'], alpha=-group['lr'] * group['alpha'])

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None):
        if closure is None:
            raise NoClosureError(str(self))

        closure = torch.enable_grad()(closure)

        enable_running_stats(self.model)
        loss = closure()

        self.first_step(zero_grad=True)

        disable_running_stats(self.model)
        closure()

        self.second_step()

        return loss

    def grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]['params'][0].device

        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class BSAM(BaseOptimizer):
    r"""SAM as an Optimal Relaxation of Bayes.

    Example:
    -------
        Here's an example::

            model = YourModel()
            optimizer = BSAM(model.parameters(), ...)

            def closure():
                loss = loss_function(output, model(input))
                loss.backward()
                return loss

            for input, output in data:
                loss = loss_function(output, model(input))
                loss.backward()

                optimizer.step(closure)
                optimizer.zero_grad()

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param num_data: int. number of training data.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param rho: float. size of the neighborhood for computing the max loss.
    :param adaptive: bool. element-wise Adaptive SAM.
    :param damping: float. damping to stabilize the method.
    :param kwargs: Dict. parameters for optimizer.
    """

    def __init__(
        self,
        params: PARAMETERS,
        num_data: int,
        lr: float = 5e-1,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 1e-4,
        rho: float = 0.05,
        adaptive: bool = False,
        damping: float = 0.1,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(num_data, 'num_data')
        self.validate_non_negative(damping, 'damping')

        self.num_data = num_data
        self.damping = damping

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'bSAM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['s'] = torch.ones_like(p)
                state['noisy_gradient'] = torch.zeros_like(p.grad)
                state['momentum'] = torch.zeros_like(p)

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if 's' not in state:
                    state['s'] = torch.ones_like(p)
                    state['noisy_gradient'] = torch.zeros_like(p.grad)
                    state['momentum'] = torch.zeros_like(p)

                noise = torch.normal(0.0, 1 / (self.num_data * state['s']))

                p.add_(noise)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['noisy_gradient'] = p.grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * group['rho'] * p.grad / state['s']

                p.add_(e_w)

    @torch.no_grad()
    def third_step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                momentum, s = state['momentum'], state['s']
                momentum.mul_(beta1).add_(p.grad * weight_decay, alpha=1.0 - beta1)

                var = (torch.sqrt(s).mul_(p.grad.abs()).add_(weight_decay + self.damping)).pow_(2)
                s.mul_(beta2).add_(var, alpha=1.0 - beta2)

                p.add_(momentum / s, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: CLOSURE = None):
        if closure is None:
            raise NoClosureError(str(self))

        self.first_step()

        with torch.enable_grad():
            closure()

        self.second_step()

        with torch.enable_grad():
            loss = closure()

        self.third_step()

        return loss
