
import torch
from torch import nn
from torch.nn.functional import cross_entropy


class LDAMLoss(nn.Module):
    """Label-distribution-aware margin loss for multiclass logits.

    Args:
        num_class_list: List of the number of samples per class.
        max_m: Maximum margin (the `C` term in the paper).
        weight: Optional class weights for reweighting.
        s: Scaling factor for logits.

    """

    def __init__(
        self, num_class_list: list[int], max_m: float = 0.5, weight: torch.Tensor | None = None, s: float = 30.0
    ):
        super().__init__()

        cls_num_list: torch.Tensor = torch.FloatTensor(num_class_list)
        m_list: torch.Tensor = 1.0 / cls_num_list.sqrt_().sqrt_()
        m_list *= max_m / max(m_list)

        self.register_buffer('m_list', m_list.unsqueeze(0))
        self.register_buffer('weight', weight)
        self.s = s

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss for predictions and targets.

        Args:
            y_pred: Class logits with shape `(N, C)`.
            y_true: Class indices with shape `(N,)`.

        Returns:
            torch.Tensor: Mean cross entropy after applying class dependent margins.

        """
        index = torch.zeros_like(y_pred, dtype=torch.bool)
        index.scatter_(1, y_true.view(-1, 1), 1)

        batch_m = torch.matmul(self.m_list.to(y_pred), index.to(dtype=y_pred.dtype).transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = y_pred - batch_m

        output = torch.where(index, x_m, y_pred)
        return cross_entropy(self.s * output, y_true, weight=self.weight)
