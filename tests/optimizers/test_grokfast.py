import pytest
import torch

from pytorch_optimizer.optimizer.grokfast import gradfilter_ema, gradfilter_ma
from tests.fixtures import TrainingModel


class TestGrokfast:
    @pytest.mark.parametrize('filter_type', ['mean', 'sum'])
    def test_grokfast_ma(self, filter_type):
        model = TrainingModel()
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        gradfilter_ma(model, None, window_size=1, filter_type=filter_type, warmup=False)

    def test_grokfast_ma_invalid(self):
        with pytest.raises(NotImplementedError):
            gradfilter_ma(TrainingModel(), None, window_size=1, filter_type='asdf', warmup=False)

    def test_grokfast_ema(self):
        model = TrainingModel()
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        grads = gradfilter_ema(model, None, alpha=0.5)
        for param in model.parameters():
            param.grad.fill_(3.0)
        gradfilter_ema(model, grads, alpha=0.5)
        for name, param in model.named_parameters():
            torch.testing.assert_close(grads[name], torch.full_like(param, 2.0))
