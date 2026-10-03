import pytest

import pytorch_optimizer
from pytorch_optimizer.loss import LOSS_FUNCTION_LIST, LOSS_FUNCTIONS, get_supported_loss_functions
from pytorch_optimizer.lr_scheduler import LR_SCHEDULER_LIST, get_supported_lr_schedulers, load_lr_scheduler
from pytorch_optimizer.optimizer import OPTIMIZER_LIST, get_supported_optimizers, load_optimizer

INVALID_OPTIMIZER_NAMES: tuple[str, ...] = (
    'asam',
    'sam',
    'gsam',
    'wsam',
    'looksam',
    'friendlysam',
    'pcgrad',
    'lookahead',
    'trac',
)


INVALID_LR_SCHEDULER_NAMES = ['dummy']


@pytest.mark.parametrize('components', [OPTIMIZER_LIST, LOSS_FUNCTION_LIST, list(LR_SCHEDULER_LIST.values())])
def test_top_level_component_exports(components):
    namespace = {}
    exec('from pytorch_optimizer import *', namespace)  # noqa: S102

    for component in components:
        assert getattr(pytorch_optimizer, component.__name__) is component
        assert namespace[component.__name__] is component

    assert not {'torch', 'fnmatch', 'Sequence', 'OptimizerType', 'ParamsT'} & namespace.keys()


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizer_invalid(invalid_optimizer_names):
    with pytest.raises(NotImplementedError):
        load_optimizer(invalid_optimizer_names)


@pytest.mark.parametrize('invalid_lr_scheduler_names', INVALID_LR_SCHEDULER_NAMES)
def test_load_lr_scheduler_invalid(invalid_lr_scheduler_names):
    with pytest.raises(NotImplementedError):
        load_lr_scheduler(invalid_lr_scheduler_names)


def test_get_supported_optimizers():
    supported = get_supported_optimizers()

    assert supported == sorted(set(supported))
    assert set(supported) == {optimizer.__name__.lower() for optimizer in OPTIMIZER_LIST}
    assert len(supported) == len(OPTIMIZER_LIST)
    for name in supported:
        assert load_optimizer(name).__name__.lower() == name

    assert get_supported_optimizers('adam*') == [name for name in supported if name.startswith('adam')]
    assert get_supported_optimizers(['adam*', 'ranger*']) == [
        name for name in supported if name.startswith(('adam', 'ranger'))
    ]
    assert get_supported_optimizers(['adam*', '*adam*']) == [name for name in supported if 'adam' in name]
    assert get_supported_optimizers('unknown*') == []


def test_get_supported_lr_schedulers():
    supported = get_supported_lr_schedulers()

    assert supported == sorted(set(supported))
    assert set(supported) == {str(name).lower() for name in LR_SCHEDULER_LIST}
    assert len(supported) == len(LR_SCHEDULER_LIST)
    for name, scheduler in LR_SCHEDULER_LIST.items():
        assert load_lr_scheduler(str(name)) is scheduler

    assert get_supported_lr_schedulers('cosine*') == [name for name in supported if name.startswith('cosine')]
    assert get_supported_lr_schedulers(['cosine*', '*warm*']) == [
        name for name in supported if name.startswith('cosine') or 'warm' in name
    ]
    assert get_supported_lr_schedulers(['cosine*', '*cosine*']) == [name for name in supported if 'cosine' in name]
    assert get_supported_lr_schedulers('unknown*') == []


def test_get_supported_loss_functions():
    supported = get_supported_loss_functions()

    assert supported == sorted(set(supported))
    assert set(supported) == {loss_function.__name__.lower() for loss_function in LOSS_FUNCTION_LIST}
    assert len(supported) == len(LOSS_FUNCTION_LIST)
    for loss_function in LOSS_FUNCTION_LIST:
        assert LOSS_FUNCTIONS[loss_function.__name__.lower()] is loss_function

    assert get_supported_loss_functions('*focal*') == [name for name in supported if 'focal' in name]
    assert get_supported_loss_functions(['*focal*', 'bce*']) == [
        name for name in supported if 'focal' in name or name.startswith('bce')
    ]
    assert get_supported_loss_functions(['focal*', '*focal*']) == [name for name in supported if 'focal' in name]
    assert get_supported_loss_functions('unknown*') == []
