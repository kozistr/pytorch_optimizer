import pytest

from pytorch_optimizer.optimizer import create_optimizer, load_optimizer
from tests.constants import SKIP_CREATE_OPTIMIZER, VALID_OPTIMIZER_NAMES
from tests.utils import Example

WRAPPER_TEST_OPTIMIZERS = ['adamp', 'lion', 'lamb', 'adan', 'madgrad', 'ranger']


def _get_optimizer_kwargs(optimizer_name):
    """Get extra kwargs required for specific optimizers."""
    kwargs = {'eps': 1e-8, 'k': 7}
    if optimizer_name == 'ranger21':
        kwargs['num_iterations'] = 1
    elif optimizer_name == 'bsam':
        kwargs['num_data'] = 1
    return kwargs


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_create_optimizer_basic(optimizer_name):
    """Test basic optimizer creation without wrappers."""
    if optimizer_name in SKIP_CREATE_OPTIMIZER:
        pytest.skip(f'skip {optimizer_name}')

    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=False,
        use_orthograd=False,
        **_get_optimizer_kwargs(optimizer_name),
    )


@pytest.mark.parametrize('optimizer_name', WRAPPER_TEST_OPTIMIZERS)
def test_create_optimizer_with_lookahead(optimizer_name):
    """Test optimizer creation with Lookahead wrapper."""
    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=True,
        use_orthograd=False,
        **_get_optimizer_kwargs(optimizer_name),
    )


@pytest.mark.parametrize('optimizer_name', WRAPPER_TEST_OPTIMIZERS)
def test_create_optimizer_with_orthograd(optimizer_name):
    """Test optimizer creation with OrthoGrad wrapper."""
    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=False,
        use_orthograd=True,
        **_get_optimizer_kwargs(optimizer_name),
    )


@pytest.mark.parametrize(
    'optimizer_name',
    [
        'bnb_adamw8bit',
        'q_galore_adamw8bit',
        'torchao_adamw4bit',
    ],
)
def test_external_optimizers_require_import(optimizer_name):
    with pytest.raises(ImportError):
        load_optimizer(optimizer_name)
