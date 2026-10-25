import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import group_tensors_by_device_and_dtype


class SWATS(BaseOptimizer):
    """Adaptive updates that switch from Adam to SGD.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        ams_bound: Use the running maximum of the second moment to bound adaptive updates.
        nesterov: Use Nesterov momentum.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch updates and phase checks. `None` enables batching on CUDA.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        nesterov: bool = False,
        eps: float = 1e-6,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'nesterov': nesterov,
            'phase': 'adam',
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SWATS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

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
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg2'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = self.apply_adam_debias(
                adam_debias=group.get('adam_debias', False),
                step_size=group['lr'] * math.sqrt(bias_correction2),
                bias_correction1=bias_correction1,
            )

            next_lr: float | None = None

            if self.can_use_foreach(
                group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
            ):
                self._step_foreach_group(group, step_size, bias_correction2)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['phase'] == 'sgd':
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']
                    buf.mul_(beta1).add_(grad)

                    update = buf.mul(1.0 - beta1)

                    if group['nesterov']:
                        update.add_(buf, alpha=beta1)

                    p.add_(update, alpha=-group['lr'])

                    continue

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )

                perturb = de_nom
                torch.div(exp_avg, de_nom, out=perturb)
                perturb.mul_(-step_size)

                p.add_(perturb)

                perturb_view = perturb.view(-1)
                pg = perturb_view.dot(grad.view(-1))

                if pg != 0:
                    scaling = perturb_view.dot(perturb_view).div_(-pg)

                    exp_avg2 = state['exp_avg2']
                    exp_avg2.lerp_(scaling, weight=1.0 - beta2)

                    corrected_exp_avg = exp_avg2 / bias_correction2

                    if (
                        group['step'] > 1
                        and corrected_exp_avg > 0.0
                        and corrected_exp_avg.allclose(scaling, rtol=group['eps'])
                    ):
                        next_lr = corrected_exp_avg.item()

            if next_lr is not None:
                group['phase'] = 'sgd'
                group['lr'] = next_lr

        return loss

    def _step_foreach_group(self, group: ParamGroup, step_size: float, bias_correction2: float) -> None:
        keys = ['exp_avg', 'exp_avg_sq', 'exp_avg2']

        if group['ams_bound']:
            keys.append('max_exp_avg_sq')

        params, grads, state = self.collect_trainable_params(group, self.state, keys)
        candidates, positions, rates = [], [], {}
        beta1, beta2 = group['betas']

        for batch in group_tensors_by_device_and_dtype(params, grads, state):
            params, grads = batch['params'], batch['grads']

            if self.maximize:
                torch._foreach_neg_(grads)

            self.apply_weight_decay_foreach(
                params, grads, group['lr'], group['weight_decay'], group['weight_decouple'], group['fixed_decay']
            )

            if group['phase'] == 'sgd':
                for p in params:
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p)

                buffers = [self.state[p]['momentum_buffer'] for p in params]
                torch._foreach_mul_(buffers, beta1)
                torch._foreach_add_(buffers, grads)
                updates = torch._foreach_mul(buffers, 1.0 - beta1)

                if group['nesterov']:
                    torch._foreach_add_(updates, buffers, alpha=beta1)

                torch._foreach_add_(params, updates, alpha=-group['lr'])
                continue

            torch._foreach_lerp_(batch['exp_avg'], grads, weight=1.0 - beta1)
            torch._foreach_mul_(batch['exp_avg_sq'], beta2)
            torch._foreach_addcmul_(batch['exp_avg_sq'], grads, grads, value=1.0 - beta2)

            denominators = self.apply_ams_bound_foreach(
                group['ams_bound'], batch['exp_avg_sq'], batch.get('max_exp_avg_sq', []), eps=group['eps']
            )

            updates = torch._foreach_div(batch['exp_avg'], denominators)
            torch._foreach_mul_(updates, -step_size)
            torch._foreach_add_(params, updates)

            pg = torch.stack([product.sum() for product in torch._foreach_mul(updates, grads)])
            squared_norms = torch.stack([product.sum() for product in torch._foreach_mul(updates, updates)])
            valid = pg != 0
            scaling = squared_norms / torch.where(valid, -pg, 1.0)

            averages = torch.stack(batch['exp_avg2']).reshape(-1)
            averages = torch.where(valid, averages.lerp(scaling, 1.0 - beta2), averages)
            torch._foreach_copy_(batch['exp_avg2'], averages.unsqueeze(1).unbind())
            corrected = averages / bias_correction2

            if group['step'] > 1:
                eligible = valid & (corrected > 0) & torch.isclose(corrected, scaling, rtol=group['eps'])
                candidates.extend(eligible.unbind())
                positions.extend(batch['indices'])
                rates.update(zip(batch['indices'], corrected.unbind()))

        if candidates:
            device = candidates[0].device
            eligible = torch.stack([candidate.to(device) for candidate in candidates])
            indices = torch.tensor(positions, device=device)
            index = int(torch.where(eligible, indices, -1).amax().item())

            if index >= 0:
                group['phase'] = 'sgd'
                group['lr'] = rates[index].item()
