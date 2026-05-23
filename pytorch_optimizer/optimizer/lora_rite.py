from typing import Any, Dict, List, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class LoRARiteHelper:
    """LoRARite Helper."""

    def __init__(self, maybe_inf_to_nan: bool = True):
        self.maybe_inf_to_nan = maybe_inf_to_nan

    def inf_to_nan(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.maybe_inf_to_nan:
            return tensor
        return torch.nan_to_num(tensor, nan=torch.nan, posinf=torch.nan, neginf=torch.nan)

    @staticmethod
    def bias_corrected_decay(step: int, decay: float) -> float:
        next_step: float = float(step + 1)
        return decay * (1.0 - decay ** (next_step - 1.0)) / (1.0 - decay**next_step)

    @staticmethod
    def move_lora_dim_to_last(tensor: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Size]:
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
            dim = 0

        moved = torch.moveaxis(tensor, dim, -1)
        return moved.reshape(-1, moved.shape[-1]), moved.shape

    @staticmethod
    def restore_original_shape_and_dim(tensor: torch.Tensor, dim: int, shape: torch.Size) -> torch.Tensor:
        return torch.moveaxis(tensor.reshape(shape), -1, dim)

    def restore_param_shape(self, tensor: torch.Tensor, param: torch.Tensor, dim: int) -> torch.Tensor:
        _, shape = self.move_lora_dim_to_last(param, dim)
        return self.restore_original_shape_and_dim(tensor, dim, shape)

    @staticmethod
    def make_symmetric(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.add(tensor.mT).mul_(0.5)

    @staticmethod
    def create_preconditioner(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((tensor.shape[-1], tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)

    def inverse_sqrt(
        self,
        tensor: torch.Tensor,
        escape: torch.Tensor,
        eps: float,
        eps_root: float,
        relative_epsilon: bool,
    ) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(self.make_symmetric(tensor))
        if relative_epsilon:
            eps_root = torch.max(eigenvalues).clamp(min=0.0).mul(eps_root).item()

        inverse_eigenvalues = eigenvalues.clamp(min=0.0).add(escape).add(eps_root).sqrt_().add_(eps).reciprocal_()
        inverse_root = eigenvectors.mul(inverse_eigenvalues.unsqueeze(0)) @ eigenvectors.mT
        return self.make_symmetric(inverse_root).to(tensor.dtype)

    def transform_second_moment_to_new_basis(self, moments: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.make_symmetric(projection @ moments @ projection.mT))

    @staticmethod
    def transform_first_moment_to_new_basis(moments: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        return moments @ projection.mT

    @staticmethod
    def get_unmagnified_grad(grad: torch.Tensor, rotate_inv: torch.Tensor) -> torch.Tensor:
        return grad @ rotate_inv

    @staticmethod
    def rotate_update(update: torch.Tensor, rotate_inv: torch.Tensor) -> torch.Tensor:
        return update @ rotate_inv.mT

    def get_unmagnified_rotate_second_escape(
        self, new_moments: torch.Tensor, old_moments: torch.Tensor
    ) -> torch.Tensor:
        old_eigenvalues = torch.linalg.eigvalsh(old_moments)
        new_eigenvalues = torch.linalg.eigvalsh(new_moments)
        zero = torch.zeros((), dtype=old_eigenvalues.dtype, device=old_eigenvalues.device)
        eigen_diff = torch.maximum(torch.max(old_eigenvalues - new_eigenvalues), zero)
        trace_diff = torch.maximum(torch.trace(old_moments) - torch.trace(new_moments), zero)
        return torch.minimum(eigen_diff, trace_diff)

    def get_preconditioned_update(
        self,
        grad: torch.Tensor,
        moments: torch.Tensor,
        escape: torch.Tensor,
        eps: float,
        eps_root: float,
        relative_epsilon: bool,
        apply_escape: bool,
    ) -> torch.Tensor:
        escape = escape if apply_escape else torch.zeros((), dtype=moments.dtype, device=moments.device)
        inverse_root = self.inverse_sqrt(moments, escape, eps, eps_root, relative_epsilon)
        return torch.nan_to_num(grad @ inverse_root)

    def update_first_moment(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta1: float
    ) -> torch.Tensor:
        beta1_decay = self.bias_corrected_decay(step, beta1)
        return update.mul(1.0 - beta1_decay).add_(moments, alpha=beta1_decay)

    def compute_second_moment(self, update: torch.Tensor) -> torch.Tensor:
        return self.make_symmetric(update.mT @ update).div_(update.shape[0])

    def update_second_moment(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta2: float
    ) -> torch.Tensor:
        beta2_decay = self.bias_corrected_decay(step, beta2)
        return update.mul(1.0 - beta2_decay).add_(moments, alpha=beta2_decay)

    def update_second_escape(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta2: float
    ) -> torch.Tensor:
        beta2_decay = self.bias_corrected_decay(step, beta2)
        return update.mul(1.0 - beta2_decay).add(moments, alpha=beta2_decay)

    @staticmethod
    def get_rotation_and_basis(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.qr(tensor)

    @staticmethod
    def reduce_rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pow(2).mean().sqrt()

    def clip_update(self, update: torch.Tensor, clip_threshold: float) -> torch.Tensor:
        update_rms = self.inf_to_nan(self.reduce_rms(update))
        return update / torch.maximum(torch.ones_like(update_rms), update_rms / clip_threshold)

    def skip_update(self, update: torch.Tensor, skip_threshold: float) -> torch.Tensor:
        update_rms = self.inf_to_nan(self.reduce_rms(update))
        if update_rms > skip_threshold:
            return torch.zeros_like(update)
        return update


class LoRARite(BaseOptimizer):
    """Robust Invariant Transformation Equilibration for LoRA optimization.

    This optimizer expects LoRA factors in alternating order, such as ``lora_a_1, lora_b_1, lora_a_2, lora_b_2``.
    Unpaired parameters and pairs with missing gradients are skipped, matching common fine-tuning workflows where only
    part of the model may receive gradients on a given step.

    Args:
        params (ParamsT): Iterable of LoRA parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for first-moment and matrix second-moment estimates.
        eps (float): Term added to the denominator to improve numerical stability.
        relative_epsilon (bool): Scale the root epsilon by the largest matrix second-moment eigenvalue.
        clip_unmagnified_grad (float): Global clipping threshold for unmagnified LoRA gradients. Disabled when 0.
        update_capping (float): Per-update RMS capping threshold after preconditioning. Disabled when 0.
        update_skipping (float): Skip unmagnified updates whose RMS is above this threshold. Disabled when 0.
        weight_decay (float): Coupled weight decay coefficient.
        apply_escape (bool): Apply the RITE escape correction when rotating second-moment bases.
        lora_l_dim (int): LoRA rank dimension for left factors.
        lora_r_dim (int): LoRA rank dimension for right factors.
        maybe_inf_to_nan (bool): Convert infinite update statistics to NaN before threshold checks.
        balance_param (bool): Balance the norms of each LoRA factor pair after applying the update.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        eps: float = 1e-6,
        relative_epsilon: bool = False,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        weight_decay: float = 0.0,
        apply_escape: bool = False,
        lora_l_dim: int = 0,
        lora_r_dim: int = -1,
        maybe_inf_to_nan: bool = True,
        balance_param: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(clip_unmagnified_grad, 'clip_unmagnified_grad')
        self.validate_non_negative(update_capping, 'update_capping')
        self.validate_non_negative(update_skipping, 'update_skipping')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.helper = LoRARiteHelper(maybe_inf_to_nan=maybe_inf_to_nan)
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'eps_root': eps**2,
            'relative_epsilon': relative_epsilon,
            'clip_unmagnified_grad': clip_unmagnified_grad,
            'update_capping': update_capping,
            'update_skipping': update_skipping,
            'weight_decay': weight_decay,
            'apply_escape': apply_escape,
            'lora_l_dim': lora_l_dim,
            'lora_r_dim': lora_r_dim,
            'balance_param': balance_param,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'LoRARite'

    @staticmethod
    def iter_lora_pairs(group: ParamGroup) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        params = list(group['params'])
        return list(zip(params[::2], params[1::2]))

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for param in group['params']:
            if param.grad is None:
                continue

            if param.grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(param):
                raise NoComplexParameterError(str(self))

    def init_pair_state(
        self, group: ParamGroup, state: Dict[str, Any], param_left: torch.Tensor, param_right: torch.Tensor
    ) -> None:
        if 'step' in state:
            return

        param_left, _ = self.helper.move_lora_dim_to_last(param_left, group['lora_l_dim'])
        param_right, _ = self.helper.move_lora_dim_to_last(param_right, group['lora_r_dim'])

        state['step'] = 0
        state['v_l'] = self.helper.create_preconditioner(param_left)
        state['v_r'] = self.helper.create_preconditioner(param_right)
        state['m_l'] = torch.zeros_like(param_left)
        state['m_r'] = torch.zeros_like(param_right)
        state['basis_l'] = torch.zeros_like(param_left)
        state['basis_r'] = torch.zeros_like(param_right)
        state['escape_l'] = torch.zeros((), dtype=param_left.dtype, device=param_left.device)
        state['escape_r'] = torch.zeros((), dtype=param_right.dtype, device=param_right.device)

    def build_pair_info(
        self,
        group: ParamGroup,
        param_left: torch.Tensor,
        param_right: torch.Tensor,
    ) -> Dict[str, Any]:
        helper = self.helper
        state = self.state[param_left]
        self.init_pair_state(group, state, param_left, param_right)

        param_left_2d, _ = helper.move_lora_dim_to_last(param_left, group['lora_l_dim'])
        param_right_2d, _ = helper.move_lora_dim_to_last(param_right, group['lora_r_dim'])

        basis_left, rotate_left = helper.get_rotation_and_basis(param_left_2d)
        basis_right, rotate_right = helper.get_rotation_and_basis(param_right_2d)
        rotate_inv_left = torch.linalg.pinv(rotate_left)
        rotate_inv_right = torch.linalg.pinv(rotate_right)

        projection_left = basis_right.mT @ state['basis_r']
        projection_right = basis_left.mT @ state['basis_l']

        grad_left = helper.inf_to_nan(param_left.grad.detach())
        grad_right = helper.inf_to_nan(param_right.grad.detach())
        if self.maximize:
            grad_left = grad_left.neg()
            grad_right = grad_right.neg()

        grad_left, _ = helper.move_lora_dim_to_last(grad_left, group['lora_l_dim'])
        grad_right, _ = helper.move_lora_dim_to_last(grad_right, group['lora_r_dim'])

        update_left = helper.get_unmagnified_grad(grad_left, rotate_inv_right)
        update_right = helper.get_unmagnified_grad(grad_right, rotate_inv_left)

        if group['update_skipping'] > 0.0:
            update_left = helper.skip_update(update_left, group['update_skipping'])
            update_right = helper.skip_update(update_right, group['update_skipping'])

        state['basis_l'] = basis_left
        state['basis_r'] = basis_right
        state['rotate_inv_l'] = rotate_inv_left
        state['rotate_inv_r'] = rotate_inv_right
        state['update_l'] = update_left
        state['update_r'] = update_right
        state['projection_l'] = projection_left
        state['projection_r'] = projection_right

        return state

    def apply_pair_update(
        self,
        group: ParamGroup,
        param_left: torch.Tensor,
        param_right: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> None:
        helper = self.helper
        state = self.state[param_left]
        update_left = state.pop('update_l')
        update_right = state.pop('update_r')
        rotate_inv_left = state.pop('rotate_inv_l')
        rotate_inv_right = state.pop('rotate_inv_r')
        projection_left = state.pop('projection_l')
        projection_right = state.pop('projection_r')
        beta1, beta2 = group['betas']

        param_left_2d, _ = helper.move_lora_dim_to_last(param_left, group['lora_l_dim'])
        param_right_2d, _ = helper.move_lora_dim_to_last(param_right, group['lora_r_dim'])

        if group['clip_unmagnified_grad'] > 0.0 and grad_norm > group['clip_unmagnified_grad']:
            scale = group['clip_unmagnified_grad'] / grad_norm
            update_left = update_left * scale
            update_right = update_right * scale

        second_left = helper.compute_second_moment(update_left)
        second_right = helper.compute_second_moment(update_right)

        transformed_v_left = helper.transform_second_moment_to_new_basis(state['v_l'], projection_left)
        transformed_v_right = helper.transform_second_moment_to_new_basis(state['v_r'], projection_right)

        if group['apply_escape']:
            escape_left = helper.get_unmagnified_rotate_second_escape(transformed_v_left, state['v_l'])
            escape_right = helper.get_unmagnified_rotate_second_escape(transformed_v_right, state['v_r'])
            escape_left = helper.update_second_escape(
                state['step'], torch.zeros_like(escape_left), escape_left + state['escape_l'], beta2
            )
            escape_right = helper.update_second_escape(
                state['step'], torch.zeros_like(escape_right), escape_right + state['escape_r'], beta2
            )
        else:
            escape_left = torch.zeros((), dtype=param_left_2d.dtype, device=param_left_2d.device)
            escape_right = torch.zeros((), dtype=param_right_2d.dtype, device=param_right_2d.device)

        v_left = helper.update_second_moment(state['step'], second_left, transformed_v_left, beta2)
        v_right = helper.update_second_moment(state['step'], second_right, transformed_v_right, beta2)

        update_left = helper.get_preconditioned_update(
            update_left,
            v_left,
            escape_left,
            group['eps'],
            group['eps_root'],
            group['relative_epsilon'],
            group['apply_escape'],
        )
        update_right = helper.get_preconditioned_update(
            update_right,
            v_right,
            escape_right,
            group['eps'],
            group['eps_root'],
            group['relative_epsilon'],
            group['apply_escape'],
        )

        m_left = helper.transform_first_moment_to_new_basis(state['m_l'], projection_left)
        m_right = helper.transform_first_moment_to_new_basis(state['m_r'], projection_right)
        m_left = helper.update_first_moment(state['step'], update_left, m_left, beta1)
        m_right = helper.update_first_moment(state['step'], update_right, m_right, beta1)

        if group['update_capping'] > 0.0:
            m_left = helper.clip_update(m_left, group['update_capping'])
            m_right = helper.clip_update(m_right, group['update_capping'])

        update_left = helper.rotate_update(m_left, rotate_inv_right)
        update_right = helper.rotate_update(m_right, rotate_inv_left)

        if group['weight_decay'] > 0.0:
            update_left = update_left.add(param_left_2d, alpha=group['weight_decay'])
            update_right = update_right.add(param_right_2d, alpha=group['weight_decay'])

        update_left = update_left.mul(-group['lr'])
        update_right = update_right.mul(-group['lr'])

        if group['balance_param']:
            left_norm = torch.linalg.norm(param_left_2d + update_left).add_(1e-6)
            right_norm = torch.linalg.norm(param_right_2d + update_right).add_(1e-6)
            balanced_norm = torch.sqrt(left_norm * right_norm)
            update_left = update_left * (balanced_norm / left_norm) + param_left_2d * (balanced_norm / left_norm - 1.0)
            update_right = update_right * (balanced_norm / right_norm) + param_right_2d * (
                balanced_norm / right_norm - 1.0
            )

        param_left.add_(helper.restore_param_shape(update_left, param_left, group['lora_l_dim']).to(param_left.dtype))
        param_right.add_(
            helper.restore_param_shape(update_right, param_right, group['lora_r_dim']).to(param_right.dtype)
        )

        state['step'] += 1
        state['v_l'] = v_left
        state['v_r'] = v_right
        state['m_l'] = m_left
        state['m_r'] = m_right
        state['escape_l'] = escape_left
        state['escape_r'] = escape_right

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        pair_infos: List[Tuple[ParamGroup, torch.Tensor, torch.Tensor]] = []
        grad_norm_sq: Optional[torch.Tensor] = None

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            for param_left, param_right in self.iter_lora_pairs(group):
                if param_left.grad is None or param_right.grad is None:
                    continue

                state = self.build_pair_info(group, param_left, param_right)
                update_left, update_right = state['update_l'], state['update_r']
                if grad_norm_sq is None:
                    grad_norm_sq = update_left.new_zeros(())

                grad_norm_sq.add_(torch.linalg.norm(update_left).pow(2).to(grad_norm_sq.device))
                grad_norm_sq.add_(torch.linalg.norm(update_right).pow(2).to(grad_norm_sq.device))
                pair_infos.append((group, param_left, param_right))

        grad_norm = torch.sqrt(grad_norm_sq) if grad_norm_sq is not None else torch.zeros(())
        for group, param_left, param_right in pair_infos:
            self.apply_pair_update(group, param_left, param_right, grad_norm)

        return loss
