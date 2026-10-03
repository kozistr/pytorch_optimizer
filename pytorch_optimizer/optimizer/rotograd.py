from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any

import torch
from torch import nn

HAS_GEOTORCH: bool = find_spec('geotorch') is not None

if HAS_GEOTORCH:
    from geotorch import orthogonal


def divide(numer: torch.Tensor, de_nom: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Divide tensors with a lower bound on the denominator magnitude."""
    return (
        torch.sign(numer)
        * torch.sign(de_nom)
        * torch.exp(torch.log(numer.abs() + eps) - torch.log(de_nom.abs() + eps))
    )


class VanillaMTL(nn.Module):
    """Multitask model with a shared backbone and task specific heads."""

    def __init__(self, backbone, heads):
        super().__init__()
        self._backbone = [backbone]
        self.heads = heads

        self.rep = None
        self.grads: list = [None for _ in range(len(heads))]

    @property
    def backbone(self):
        return self._backbone[0]

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        self.backbone.train(mode)
        for head in self.heads:
            head.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _hook(self, index):
        def _hook_(g):
            self.grads[index] = g

        return _hook_

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)

        if isinstance(out, (list, tuple)):
            rep, extra_out = out[0], out[1:]
            extra_out = list(extra_out)
        else:
            rep = out
            extra_out = []

        if self.training:
            self.rep = rep

        preds: list[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            rep_i = rep
            if self.training:
                rep_i = rep.detach().clone()
                rep_i.requires_grad = True
                rep_i.register_hook(self._hook(i))

            out_i = head(rep_i)
            if isinstance(out_i, (list, tuple)):
                preds.append(out_i[0])
                extra_out.append(out_i[1:])
            else:
                preds.append(out_i)

        return preds if len(extra_out) == 0 else (preds, extra_out)

    def backward(self, losses, backbone_loss=None, **kwargs):
        for loss in losses:
            loss.backward(**kwargs)

        if backbone_loss is not None:
            backbone_loss.backward(retain_graph=True)

        self.rep.backward(sum(self.grads))

    def mtl_parameters(self, recurse=True):
        return self.parameters(recurse=recurse)

    def model_parameters(self, recurse=True):
        for param in self.backbone.parameters(recurse=recurse):
            yield param

        for h in self.heads:
            for param in h.parameters(recurse=recurse):
                yield param


def rotate(points: torch.Tensor, rotation: torch.Tensor, total_size: int) -> torch.Tensor:
    """Apply a rotation matrix to the latent representation."""
    if total_size != points.size(-1):
        points_lo, points_hi = points[:, : rotation.size(1)], points[:, rotation.size(1) :]
        point_lo = torch.einsum('ij,bj->bi', rotation, points_lo)
        return torch.cat((point_lo, points_hi), dim=-1)
    return torch.einsum('ij,bj->bi', rotation, points)


def rotate_back(points: torch.Tensor, rotation: torch.Tensor, total_size: int) -> torch.Tensor:
    """Apply the inverse rotation to the latent representation."""
    return rotate(points, rotation.t(), total_size)


class RotateModule(nn.Module):
    """Apply the learned rotation for one task."""

    def __init__(self, parent, item):
        super().__init__()

        self.parent = [parent]
        self.item = item

    def hook(self, grad: torch.Tensor):
        self.p.grads[self.item] = grad.clone()

    @property
    def p(self):
        return self.parent[0]

    @property
    def r(self):
        return self.p.rotation[self.item]

    @property
    def weight(self):
        return self.p.weight[self.item] if hasattr(self.p, 'weight') else 1.0

    def rotate(self, z: torch.Tensor) -> torch.Tensor:
        return rotate(z, self.r, self.p.latent_size)

    def rotate_back(self, z: torch.Tensor) -> torch.Tensor:
        return rotate_back(z, self.r, self.p.latent_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = self.r.clone().detach()
        new_z = rotate(z, r, self.p.latent_size)
        if self.p.training:
            new_z.register_hook(self.hook)
        return new_z


class RotateOnly(nn.Module):
    """Learn task specific rotations of a shared multitask representation.

    Args:
        backbone: Shared model producing the latent representation.
        heads: Task specific models consuming the latent representation.
        latent_size: Number of features in the shared representation.
        *args (tuple): Additional positional arguments, accepted without effect.
        burn_in_period: Steps before refreshing the reference task losses.
        normalize_losses: Normalize losses when computing gradients for the task specific heads.

    """

    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: torch.Tensor | None

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__()
        if not HAS_GEOTORCH:
            raise ImportError('[-] you need to install `geotorch` to use RotoGrad. `pip install geotorch`')

        self._backbone = [backbone]
        self.heads = heads

        self.num_tasks: int = len(heads)
        self.latent_size = latent_size
        self.burn_in_period = burn_in_period
        self.normalize_losses = normalize_losses

        for i in range(self.num_tasks):
            heads[i] = nn.Sequential(RotateModule(self, i), heads[i])

        for i in range(self.num_tasks):
            self.register_parameter(f'rotation_{i}', nn.Parameter(torch.eye(latent_size), requires_grad=True))
            orthogonal(self, f'rotation_{i}', triv='expm')  # uses exponential map (alternative: cayley)

        self.rep = None
        self.grads = [None for _ in range(self.num_tasks)]
        self.original_grads = [None for _ in range(self.num_tasks)]
        self.losses = [None for _ in range(self.num_tasks)]
        self.initial_losses = [None for _ in range(self.num_tasks)]
        self.initial_backbone_loss = None
        self.iteration_counter: int = 0

    @property
    def rotation(self) -> Sequence[torch.Tensor]:
        """Return trainable rotation matrices, one per task. Detach them for inspection."""
        return [getattr(self, f'rotation_{i}') for i in range(self.num_tasks)]

    @property
    def backbone(self) -> nn.Module:
        return self._backbone[0]

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        self.backbone.train(mode)
        for head in self.heads:
            head.train(mode)
        return self

    def __len__(self) -> int:
        """Get the number of tasks."""
        return self.num_tasks

    def __getitem__(self, item) -> nn.Module:
        """Get an end to end model for the selected task."""
        return nn.Sequential(self.backbone, self.heads[item])

    def _hook(self, index):
        def _hook_(g):
            self.original_grads[index] = g

        return _hook_

    def forward(self, x: Any) -> Sequence[Any]:
        """Run the shared backbone and return predictions from each task head."""
        out = self.backbone(x)

        if isinstance(out, (list, tuple)):
            rep, extra_out = out[0], out[1:]
            extra_out = list(extra_out)
        else:
            rep = out
            extra_out = []

        if self.training:
            self.rep = rep

        preds = []
        for i, head in enumerate(self.heads):
            rep_i = rep
            if self.training:
                rep_i = rep.detach().clone()
                rep_i.requires_grad = True
                rep_i.register_hook(self._hook(i))

            out_i = head(rep_i)
            if isinstance(out_i, (list, tuple)):
                preds.append(out_i[0])
                extra_out.append(out_i[1:])
            else:
                preds.append(out_i)

        return preds if len(extra_out) == 0 else (preds, extra_out)

    def backward(self, losses: Sequence[torch.Tensor], backbone_loss=None, **kwargs) -> None:
        """Backpropagate task losses and compute gradients for the rotation matrices.

        Args:
            losses: Scalar loss for each task.
            backbone_loss (torch.Tensor | None): Optional additional loss for the shared representation.
            **kwargs (dict): Options for `torch.Tensor.backward`.

        """
        if not self.training:
            raise AssertionError('Backward should only be called when training')

        if self.iteration_counter in (0, self.burn_in_period):
            for i, loss in enumerate(losses):
                self.initial_losses[i] = loss.item()

            if self.normalize_losses and backbone_loss is not None:
                self.initial_backbone_loss = backbone_loss.item()

        self.iteration_counter += 1

        for i in range(len(losses)):
            loss = losses[i] / self.initial_losses[i]
            self.losses[i] = loss.item()

            if self.normalize_losses:
                loss.backward(**kwargs)
            else:
                losses[i].backward(**kwargs)

        if backbone_loss is not None:
            if self.normalize_losses:
                (backbone_loss / self.initial_backbone_loss).backward(retain_graph=True)
            else:
                backbone_loss.backward(retain_graph=True)

        self.rep.backward(self._rep_grad())

    def _rep_grad(self):
        mean_grad = sum(self.original_grads) / len(self.grads)
        mean_norm = torch.linalg.norm(mean_grad)

        mean_grad = sum(g * divide(mean_norm, torch.linalg.norm(g)) for g in self.original_grads) / len(self.grads)

        for rotation, grad in zip(self.rotation, self.grads):
            loss = rotate(mean_grad, rotation, self.latent_size) - grad
            loss = torch.einsum('bi,bi->b', loss, loss)
            loss.mean().backward()

        return sum(self.original_grads)

    def mtl_parameters(self, recurse: bool = True):
        return self.parameters(recurse=recurse)

    def model_parameters(self, recurse=True):
        for param in self.backbone.parameters(recurse=recurse):
            yield param

        for h in self.heads:
            for param in h.parameters(recurse=recurse):
                yield param


class RotoGrad(RotateOnly):
    """Balance multitask gradient directions and magnitudes with learned rotations.

    Args:
        backbone: Shared model producing the latent representation.
        heads: Task specific models consuming the latent representation.
        latent_size: Number of features in the shared representation.
        *args (tuple): Additional positional arguments, accepted without effect.
        burn_in_period: Ignored. This variant uses the base default of 20 steps.
        normalize_losses: Normalize losses when computing gradients for the task specific heads.

    """

    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__(backbone, heads, latent_size, burn_in_period, *args, normalize_losses=normalize_losses)

        self.initial_grads = None
        self.counter: int = 0

    def _rep_grad(self):
        super()._rep_grad()

        grad_norms = [torch.linalg.norm(g, keepdim=True).clamp_min(1e-15) for g in self.original_grads]
        if self.initial_grads is None or self.counter == self.burn_in_period:
            self.initial_grads = grad_norms
            conv_ratios = [torch.ones((1,)) for _ in range(len(self.initial_grads))]
        else:
            conv_ratios = [x / y for x, y in zip(grad_norms, self.initial_grads)]

        self.counter += 1

        alphas = [x / torch.clamp(sum(conv_ratios), 1e-15) for x in conv_ratios]
        weighted_sum_norms = sum(a * g for a, g in zip(alphas, grad_norms))

        return sum(g / n * weighted_sum_norms for g, n in zip(self.original_grads, grad_norms))


class RotoGradNorm(RotoGrad):
    """Balance rotated multitask gradients with GradNorm task weights.

    Args:
        backbone: Shared model producing the latent representation.
        heads: Task specific models consuming the latent representation.
        latent_size: Number of features in the shared representation.
        *args (tuple): Additional positional arguments, accepted without effect.
        alpha: Exponent controlling the GradNorm target training rates.
        burn_in_period: Ignored. This variant uses the base default of 20 steps.
        normalize_losses: Normalize losses when computing gradients for the task specific heads.

    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        alpha: float,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__(
            backbone, heads, latent_size, *args, burn_in_period=burn_in_period, normalize_losses=normalize_losses
        )
        self.alpha = alpha
        self.weight_ = nn.ParameterList([nn.Parameter(torch.ones([]), requires_grad=True) for _ in range(len(heads))])

    @property
    def weight(self) -> Sequence[torch.Tensor]:
        """Return normalized positive task weights. Detach them for inspection."""
        ws = [w.exp() + 1e-15 for w in self.weight_]
        norm_coef = self.num_tasks / sum(ws)
        return [w * norm_coef for w in ws]

    def _rep_grad(self):
        super()._rep_grad()

        grads_norm = [torch.linalg.norm(g) for g in self.original_grads]

        mean_grad = sum(g * w for g, w in zip(self.original_grads, self.weight)) / len(self.grads)

        mean_grad_norm = torch.linalg.norm(mean_grad)
        mean_loss = sum(self.losses) / len(self.losses)

        for i, (loss, grad) in enumerate(zip(self.losses, grads_norm)):
            inverse_ratio_i = (loss / mean_loss) ** self.alpha
            mean_grad_i = mean_grad_norm * float(inverse_ratio_i)

            loss_grad_norm = torch.abs(grad * self.weight[i] - mean_grad_i)
            loss_grad_norm.backward()

        with torch.no_grad():
            return sum(g * w for g, w in zip(self.original_grads, self.weight))
