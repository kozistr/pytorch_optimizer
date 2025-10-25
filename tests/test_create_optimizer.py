import pytest

from pytorch_optimizer.optimizer import create_optimizer, load_optimizer
from tests.constants import VALID_OPTIMIZER_NAMES
from tests.utils import Example

SKIPPED_OPTIMIZERS = {'adamw', 'adam', 'sgd', 'lbfgs', 'nadam', 'rmsprop', 'demo', 'distributedmuon'}


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
@pytest.mark.parametrize('use_lookahead', [True, False])
@pytest.mark.parametrize('use_orthograd', [True, False])
def test_create_optimizer(optimizer_name, use_lookahead, use_orthograd):
    if optimizer_name in SKIPPED_OPTIMIZERS or (use_lookahead and use_orthograd):
        pytest.skip(f'skip {optimizer_name} ({use_lookahead}, {use_orthograd})')

    kwargs = {'eps': 1e-8, 'k': 7}
    if optimizer_name == 'ranger21':
        kwargs.update({'num_iterations': 1})
    elif optimizer_name == 'bsam':
        kwargs.update({'num_data': 1})

    create_optimizer(
        Example(),
        optimizer_name=optimizer_name,
        use_lookahead=use_lookahead,
        use_orthograd=use_orthograd,
        **kwargs,
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
