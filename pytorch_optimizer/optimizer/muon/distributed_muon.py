import math

import torch
from torch.distributed import all_gather, get_rank, get_world_size

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon.muon import Muon
from pytorch_optimizer.optimizer.utils.matrix import NewtonSchulzWeights


class DistributedMuon(Muon):  # pragma: no cover
    """Distributed momentum updates with Newton-Schulz matrix orthogonalization.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.
    Requires an initialized distributed process group.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        use_adjusted_lr: Scale orthogonal updates using the Moonlight shape adjustment.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch updates owned by each rank. `None` enables batching on CUDA.

    Examples:
        ```python
        from pytorch_optimizer import DistributedMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = DistributedMuon(param_groups)
        ```

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            list(params) if params is not None else params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
            use_adjusted_lr=use_adjusted_lr,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
            maximize=maximize,
            foreach=foreach,
            **kwargs,
        )

        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

    def __str__(self) -> str:
        return 'DistributedMuon'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0 and not group['use_muon']:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            batched = self.can_use_foreach(group, group.get('foreach'), default_on_cuda=True)

            if batched:
                owned = group['params'][self.rank::self.world_size] if group['use_muon'] else group['params']

                for p in owned:
                    if group['use_muon'] and not self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p)

                self._step_foreach_group({**group, 'params': owned})

                if not group['use_muon']:
                    continue

            if group['use_muon']:
                params = group['params']

                if not params:
                    continue

                padded_params = params + [torch.empty_like(params[-1]) for _ in range(-len(params) % self.world_size)]

                for i in range(len(params))[:: self.world_size]:
                    if not batched and i + self.rank < len(params):
                        p = params[i + self.rank]

                        grad = p.grad

                        self.maximize_gradient(grad, maximize=self.maximize)

                        state = self.state[p]
                        if len(state) == 0:
                            state['momentum_buffer'] = torch.zeros_like(p)

                        self.apply_weight_decay(
                            p,
                            grad=grad,
                            lr=group['lr'],
                            weight_decay=group['weight_decay'],
                            weight_decouple=group['weight_decouple'],
                            fixed_decay=False,
                        )

                        buf = state['momentum_buffer']
                        update = self._momentum_updates(group, [grad], [buf])[0]
                        self._apply_muon_updates(group, [p], [grad], [update])

                    all_gather(padded_params[i:i + self.world_size], padded_params[i + self.rank])  # fmt: skip
            else:
                for p in group['params']:
                    grad = p.grad

                    self.maximize_gradient(grad, maximize=self.maximize)

                    self.apply_weight_decay(
                        p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=False,
                    )

                    state = self.state[p]
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                    p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)

        return loss
