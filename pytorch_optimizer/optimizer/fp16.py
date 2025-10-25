from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.type import Closure, Parameters
from pytorch_optimizer.optimizer.utils import clip_grad_norm, has_overflow


class DynamicLossScaler:
    r"""Dynamically adjusts the loss scaling factor.

        Dynamic loss scalers are important in mixed-precision training.
        They help us avoid underflows and overflows in low-precision gradients.

        See here for information:
        <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling>

        Shamelessly stolen and adapted from FairSeq.
        <https://github.com/pytorch/fairseq/blob/main/fairseq/optim/fp16_optimizer.py>

        Reference : 'https://github.com/facebookresearch/ParlAI/blob/main/parlai/utils/fp16.py'

    :param init_scale: Initial loss scale.
    :param scale_factor: Factor by which to increase or decrease loss scale.
    :param scale_window: If we do not experience overflow in scale_window iterations,
        loss scale will increase by scale_factor.
    :param tolerance: Pct of iterations that have overflowed after which we must decrease the loss scale.
    :param threshold: If not None, loss scale will decrease below this threshold.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        tolerance: float = 0.00,
        threshold: Optional[float] = None,
    ):  # fmt: skip
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold

        self.iter: int = 0
        self.last_overflow_iter: int = -1
        self.last_rescale_iter: int = -1
        self.overflows_since_rescale: int = 0
        self.has_overflow_serial: bool = False

    def update_scale(self, overflow: bool):
        r"""Update the loss scale.

            If overflow exceeds our tolerance, we decrease the loss scale.
            If the number of iterations since the last overflow exceeds the scale window, we increase the loss scale.

        :param overflow: bool. adjust scales to prevent overflow.
        """
        iter_since_rescale: int = self.iter - self.last_rescale_iter

        if overflow:
            # calculate how often we overflowed already
            self.last_overflow_iter = self.iter
            self.overflows_since_rescale += 1

            pct_overflow: float = self.overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                # decrease loss scale by the scale factor
                self.decrease_loss_scale()

                # reset iterations
                self.last_rescale_iter = self.iter
                self.overflows_since_rescale = 0
        elif (self.iter - self.last_overflow_iter) % self.scale_window == 0:
            # increase the loss scale by scale factor
            self.loss_scale *= self.scale_factor
            self.last_rescale_iter = self.iter

        self.iter += 1

    def decrease_loss_scale(self):
        r"""Decrease the loss scale by self.scale_factor.

        NOTE: the loss_scale will not go below `self.threshold`.
        """
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)


class SafeFP16Optimizer(Optimizer):  # pragma: no cover
    r"""Safe FP16 Optimizer.

    :param optimizer: OPTIMIZER.
    :param aggregate_g_norms: bool. aggregate_g_norms.
    :param min_loss_scale: float. min_loss_scale.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        aggregate_g_norms: bool = False,
        min_loss_scale: float = 2 ** -5,
    ):  # fmt: skip
        self.optimizer = optimizer
        self.aggregate_g_norms = aggregate_g_norms
        self.min_loss_scale = min_loss_scale

        self.fp16_params = self.get_parameters(optimizer)
        self.fp32_params = self.build_fp32_params(self.fp16_params, flatten=False)

        # we want the optimizer to be tracking the fp32 parameters
        if len(optimizer.param_groups) != 1:
            # future implementers: this should hopefully be a matter of just iterating through the param groups and
            # keeping track of the pointer through the fp32_params
            raise NotImplementedError('Need to implement the parameter group transfer.')

        optimizer.param_groups[0]['params'] = self.fp32_params

        self.scaler: DynamicLossScaler = DynamicLossScaler(2.0 ** 15)  # fmt: skip
        self.needs_sync: bool = True

    @classmethod
    def get_parameters(cls, optimizer: Optimizer) -> List:
        params: List = []
        for group in optimizer.param_groups:
            params += list(group['params'])
        return params

    @classmethod
    def build_fp32_params(
        cls, parameters: Parameters, flatten: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if flatten:
            total_param_size: int = sum(p.numel() for p in parameters)
            fp32_params = torch.zeros(total_param_size, dtype=torch.float, device=parameters[0].device)

            offset: int = 0
            for p in parameters:
                p_num_el = p.numel()
                fp32_params[offset:offset + p_num_el].copy_(p.view(-1))  # fmt: skip
                offset += p_num_el

            fp32_params = nn.Parameter(fp32_params)
            fp32_params.grad = fp32_params.new(total_param_size)

            return fp32_params

        fp32_params: List = []
        for p in parameters:
            p32 = nn.Parameter(p.float())
            p32.grad = torch.zeros_like(p32)
            fp32_params.append(p32)

        return fp32_params

    def state_dict(self) -> Dict:
        r"""Return the optimizer state dict."""
        state_dict = self.optimizer.state_dict()
        if self.scaler is not None:
            state_dict['loss_scaler'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        r"""Load an optimizer state dict.

            In general, we should prefer the configuration of the existing optimizer instance (e.g., learning rate)
            over that found in the state_dict. This allows us to resume training from a checkpoint using a new set of
            optimizer args.

        :param state_dict: Dict. state_dict.
        """
        if 'loss_scaler' in state_dict and self.scaler is not None and isinstance(state_dict['loss_scaler'], float):
            self.scaler.loss_scale = state_dict['loss_scaler']
        self.optimizer.load_state_dict(state_dict)

    def backward(self, loss, update_main_grads: bool = False):
        r"""Compute the sum of gradients of the given tensor w.r.t. graph leaves.

            Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this function
            additionally dynamically scales the loss to avoid gradient underflow.

        :param loss: float. loss.
        :param update_main_grads: bool. update main gradient.
        """
        if self.scaler is not None:
            loss = loss * self.scaler.loss_scale

        loss.backward()

        self.needs_sync = True
        if update_main_grads:
            self.update_main_grads()

    def sync_fp16_grads_to_fp32(self, multiply_grads: float = 1.0) -> None:
        r"""Sync fp16 to fp32 gradients."""
        if self.needs_sync:
            if self.scaler is not None:
                multiply_grads /= self.scaler.loss_scale

            for p16, p32 in zip(self.fp16_params, self.fp32_params):
                if not p16.requires_grad:
                    continue

                if p16.grad is not None:
                    p32.grad.copy_(p16.grad)
                    p32.grad.mul_(multiply_grads)
                else:
                    p32.grad = torch.zeros_like(p16, dtype=torch.float)

            self.needs_sync = False

    def multiply_grads(self, c: float) -> None:
        r"""Multiply grads by a constant c."""
        if self.needs_sync:
            self.sync_fp16_grads_to_fp32(c)
            return

        for p32 in self.fp32_params:
            p32.grad.mul_(c)

    def update_main_grads(self) -> None:
        self.sync_fp16_grads_to_fp32()

    def clip_main_grads(self, max_norm: float):
        r"""Clip gradient norm and updates dynamic loss scaler."""
        self.sync_fp16_grads_to_fp32()

        grad_norm = clip_grad_norm(self.fp32_params, max_norm, sync=self.aggregate_g_norms)

        # detect overflow and adjust loss scale
        if self.scaler is not None:
            overflow: bool = has_overflow(grad_norm)
            prev_scale: float = self.scaler.loss_scale

            self.scaler.update_scale(overflow)

            if overflow:
                self.zero_grad()
                if self.scaler.loss_scale <= self.min_loss_scale:
                    # Use FloatingPointError as an uncommon error
                    # that parent functions can safely catch to stop training.
                    self.scaler.loss_scale = prev_scale

                    raise FloatingPointError(
                        f'Minimum loss scale reached ({self.min_loss_scale}). Your loss is probably exploding. '
                        'Try lowering the learning rate, using gradient clipping or increasing the batch size.\n'
                        f'Overflow: setting loss scale to {self.scaler.loss_scale}'
                    )

        return grad_norm

    def step(self, closure: Closure = None):
        r"""Perform a single optimization step."""
        self.sync_fp16_grads_to_fp32()
        self.optimizer.step(closure)

        for p16, p32 in zip(self.fp16_params, self.fp32_params):
            if not p16.requires_grad:
                continue
            p16.data.copy_(p32)

    def zero_grad(self) -> None:
        r"""Clear the gradients of all optimized parameters."""
        for p16 in self.fp16_params:
            p16.grad = None
        for p32 in self.fp32_params:
            p32.grad.zero_()
        self.needs_sync = False

    def get_lr(self) -> float:
        r"""Get learning rate."""
        return self.optimizer.get_lr()

    def set_lr(self, lr: float):
        r"""Set learning rate."""
        self.optimizer.set_lr(lr)

    @property
    def loss_scale(self) -> float:
        r"""Convenience function which TorchAgent calls to get current scale value."""
        return self.scaler.loss_scale
