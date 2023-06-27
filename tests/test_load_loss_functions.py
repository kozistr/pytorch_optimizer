from pytorch_optimizer import get_supported_loss_functions


def test_get_supported_loss_functions():
    assert len(get_supported_loss_functions()) == 8
