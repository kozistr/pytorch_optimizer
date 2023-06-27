from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy


class LDAMLoss(nn.Module):
    r"""LDAM Loss.

    :param num_class_list: List[int]. list of number of class.
    :param max_m: float. max margin (`C` term in the paper).
    :param weight: Optional[torch.Tensor]. class weight.
    :param s: float. scaler.
    """

    def __init__(
        self, num_class_list: List[int], max_m: float = 0.5, weight: Optional[torch.Tensor] = None, s: float = 30.0
    ):
        super().__init__()

        cls_num_list = np.asarray(num_class_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list *= max_m / np.max(m_list)

        self.m_list = torch.FloatTensor(m_list).unsqueeze(0)
        self.weight = weight
        self.s = s

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(y_pred, dtype=torch.bool)
        index.scatter_(1, y_true.view(-1, 1), 1)

        batch_m = torch.matmul(self.m_list.to(index.device), index.float().transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = y_pred - batch_m

        output = torch.where(index, x_m, y_pred)
        return cross_entropy(self.s * output, y_true, weight=self.weight)
