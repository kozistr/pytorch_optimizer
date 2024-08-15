from typing import Callable, Dict, List, Tuple

import torch
from torch import nn

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, OPTIMIZER


def polyval(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    r"""Implement of the Horner scheme to evaluate a polynomial.

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124

    :param x: torch.Tensor. variable.
    :param coef: torch.Tensor. coefficients of the polynomial.
    """
    result = coef[0].clone()

    for c in coef[1:]:
        result = (result * x) + c

    return result[0]


class ERF1994(nn.Module):
    r"""Implementation of ERF1994.

    :param num_coefs: int. The number of polynomial coefficients to use in the approximation.
    """

    def __init__(self, num_coefs: int = 128) -> None:
        super().__init__()

        self.n: int = num_coefs

        self.i: torch.Tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        self.m = 2 * self.n
        self.m2 = 2 * self.m
        self.k = torch.linspace(-self.m + 1, self.m - 1, self.m2 - 1)
        self.l = torch.sqrt(self.n / torch.sqrt(torch.tensor(2.0)))
        self.theta = self.k * torch.pi / self.m
        self.t = self.l * torch.tan(self.theta / 2.0)
        self.f = torch.exp(-self.t ** 2) * (self.l ** 2 + self.t ** 2)  # fmt: skip
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real / self.m2
        self.a = torch.flipud(self.a[1:self.n + 1])  # fmt: skip

    def w_algorithm(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute the Faddeeva function of a complex number.

        :param z: torch.Tensor. A tensor of complex numbers.
        """
        self.l = self.l.to(z.device)
        self.i = self.i.to(z.device)
        self.a = self.a.to(z.device)

        iz = self.i * z
        lp_iz, ln_iz = self.l + iz, self.l - iz

        z_ = lp_iz / ln_iz
        p = polyval(z_.unsqueeze(0), self.a)
        return 2 * p / ln_iz.pow(2) + (1.0 / torch.sqrt(torch.tensor(torch.pi))) / ln_iz

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute the error function of a complex number.

        :param z: torch.Tensor. A tensor of complex numbers.
        """
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z * self.i)) - z ** 2) + 1  # fmt: skip
        return torch.complex(out.real * sign_r, out.imag * sign_i)


class TRAC(BaseOptimizer):
    r"""A Parameter-Free Optimizer for Lifelong Reinforcement Learning.

    Example:
    -------
        Here's an example::

            model = YourModel()
            optimizer = TRAC(AdamW(model.parameters()))

            for input, output in data:
                optimizer.zero_grad()

                loss = loss_fn(model(input), output)
                loss.backward()

                optimizer.step()

    :param optimizer: Optimizer. base optimizer.
    :param betas: List[float]. list of beta values.
    :param num_coefs: int. the number of polynomial coefficients to use in the approximation.
    :param s_prev: float. initial scale value.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        optimizer: OPTIMIZER,
        betas: List[float] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
        num_coefs: int = 128,
        s_prev: float = 1e-8,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_positive(num_coefs, 'num_coefs')
        self.validate_non_negative(s_prev, 's_prev')
        self.validate_non_negative(eps, 'eps')

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.erf = ERF1994(num_coefs=num_coefs)
        self.betas = betas
        self.s_prev = s_prev
        self.eps = eps

        self.f_term = self.s_prev / self.erf_imag(1.0 / torch.sqrt(torch.tensor(2.0)))

        self.optimizer = optimizer
        self.defaults: DEFAULTS = optimizer.defaults

    def __str__(self) -> str:
        return 'TRAC'

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @torch.no_grad()
    def reset(self):
        device = self.param_groups[0]['params'][0].device

        self.state['trac'] = {
            'betas': torch.tensor(self.betas, device=device),
            's': torch.zeros(1, device=device),
            'variance': torch.zeros(len(self.betas), device=device),
            'sigma': torch.full((len(self.betas),), 1e-8, device=device),
            'step': 0,
        }

        for group in self.param_groups:
            for p in group['params']:
                self.state['trac'][p] = p.clone()

    @torch.no_grad()
    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def erf_imag(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)

        ix = torch.complex(torch.zeros_like(x), x)

        return self.erf(ix).imag

    @torch.no_grad()
    def backup_params_and_grads(self) -> Tuple[Dict, Dict]:
        updates, grads = {}, {}

        for group in self.param_groups:
            for p in group['params']:
                updates[p] = p.clone()
                grads[p] = p.grad.clone() if p.grad is not None else None

        return updates, grads

    @torch.no_grad()
    def trac_step(self, updates: Dict, grads: Dict) -> None:
        self.state['trac']['step'] += 1

        deltas = {}

        device = self.param_groups[0]['params'][0].device

        s = self.state['trac']['s']
        h = torch.zeros((1,), device=device)
        for group in self.param_groups:
            for p in group['params']:
                if grads[p] is None:
                    continue

                theta_ref = self.state['trac'][p]
                update = updates[p]

                deltas[p] = (update - theta_ref) / s.add(self.eps)
                update.neg_().add_(p)

                grad, delta = grads[p], deltas[p]

                product = torch.dot(delta.flatten(), grad.flatten())
                h.add_(product)

                delta.add_(update)

                p.copy_(theta_ref)

        betas = self.state['trac']['betas']
        variance = self.state['trac']['variance']
        sigma = self.state['trac']['sigma']

        variance.mul_(betas.pow(2)).add_(h.pow(2))
        sigma.mul_(betas).sub_(h)

        term = self.erf_imag(sigma / (2.0 * variance).sqrt_().add_(self.eps)).mul_(self.f_term)
        s.copy_(torch.sum(term))

        scale = max(s, 0.0)

        for group in self.param_groups:
            for p in group['params']:
                if grads[p] is None:
                    continue

                p.add_(deltas[p] * scale)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        # TODO: backup is first to get the delta of param and grad, but it does not work.
        with torch.enable_grad():
            loss = self.optimizer.step(closure)

        updates, grads = self.backup_params_and_grads()

        if 'trac' not in self.state:
            device = self.param_groups[0]['params'][0].device

            self.state['trac'] = {
                'betas': torch.tensor(self.betas, device=device),
                's': torch.zeros(1, device=device),
                'variance': torch.zeros(len(self.betas), device=device),
                'sigma': torch.full((len(self.betas),), 1e-8, device=device),
                'step': 0,
            }

            for group in self.param_groups:
                for p in group['params']:
                    self.state['trac'][p] = updates[p].clone()

        self.trac_step(updates, grads)

        return loss
