from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestRanger21:
    """Tests for Ranger21 optimizer specific functionality."""

    def test_warm_iterations(self):
        assert load_optimizer('ranger21').build_warm_up_iterations(1000, 0.999) == 220
        assert load_optimizer('ranger21').build_warm_up_iterations(4500, 0.999) == 2000
        assert load_optimizer('ranger21').build_warm_down_iterations(1000) == 280

    def test_warm_up_and_down(self):
        lr: float = 1e-1
        opt = build_optimizer(
            'ranger21', [make_parameter(requires_grad=False)], num_iterations=500, lr=lr, warm_down_min_lr=3e-5
        )

        assert opt.warm_up_dampening(lr, 100) == 0.09090909090909091
        assert opt.warm_up_dampening(lr, 200) == 0.1
        assert opt.warm_up_dampening(lr, 300) == 0.1
        assert opt.warm_down(lr, 300) == 0.1
        assert opt.warm_down(lr, 400) == 0.07093070921985817

    def test_closure(self):
        param = make_parameter()
        optimizer = build_optimizer('ranger21', [param], num_iterations=100, betas=(0.9, 1e-9))

        def closure():
            loss = param.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
