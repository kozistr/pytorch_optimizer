import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer.grokfast import gradfilter_ema, gradfilter_ma


class TestGrokfast:
    @pytest.mark.parametrize('filter_type', ['mean', 'sum'])
    def test_grokfast_ma(self, filter_type):
        model = nn.Linear(1, 1)
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        gradfilter_ma(model, None, window_size=1, filter_type=filter_type, warmup=False)

    def test_grokfast_ma_invalid(self):
        with pytest.raises(NotImplementedError):
            gradfilter_ma(nn.Linear(1, 1), None, window_size=1, filter_type='asdf', warmup=False)

    def test_grokfast_ema(self):
        model = nn.Linear(1, 1)
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        gradfilter_ema(model, None)
