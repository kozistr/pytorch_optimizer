import pytest
import torch

from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from tests.fixtures import TrainingModel, build_model, make_parameter
from tests.optimizer_cases import COMPLEX_OPTIMIZERS, FOREACH_OPTIMIZERS, MODEL_OPTIMIZERS, SKIP_BF16_OPTIMIZERS
from tests.recipes import COMPILE_SUPPORTED_OPTIMIZERS, OPTIMIZER_RECIPES
from tests.utils import (
    Trainer,
    build_optimizer,
    build_optimizer_parameters,
    dummy_closure,
    ids,
    make_closure,
    should_use_create_graph,
)

TRAINING_CASES = [
    pytest.param(recipe, dtype, foreach, id=f'{ids(recipe)}-{dtype}-{foreach}')
    for recipe in OPTIMIZER_RECIPES
    for dtype in (torch.float32, torch.bfloat16, torch.complex64)
    if not (dtype == torch.float32 and recipe[0] == 'nero' and 'constraints' not in recipe[1])
    if not (dtype == torch.bfloat16 and recipe[0] in SKIP_BF16_OPTIMIZERS)
    if dtype != torch.complex64 or recipe[0] in COMPLEX_OPTIMIZERS
    for foreach in ([False, True] if dtype != torch.complex64 and recipe[0] in FOREACH_OPTIMIZERS else [False])
]
RECIPE_OPTIMIZER_NAMES = sorted({recipe[0] for recipe in OPTIMIZER_RECIPES})


class TestOptimizerTraining:
    @pytest.mark.parametrize(('optimizer_config', 'dtype', 'foreach'), TRAINING_CASES)
    def test_training(self, optimizer_config, dtype, foreach, environment):
        optimizer_name, config, iterations = optimizer_config

        x_data, y_data = environment
        model, loss_fn = build_model(use_complex=dtype == torch.complex64, device=x_data.device)
        if dtype == torch.complex64:
            x_data = x_data.to(dtype=dtype)
        else:
            model = model.to(dtype=dtype)

        parameters = model if optimizer_name in MODEL_OPTIMIZERS else model.parameters()
        parameters, config = build_optimizer_parameters(parameters, optimizer_name, config)
        if optimizer_name in FOREACH_OPTIMIZERS:
            config['foreach'] = foreach

        optimizer = build_optimizer(optimizer_name, parameters, **config)
        if optimizer_name == 'schedulefree':
            optimizer.train()

        def closure_fn(loss):
            return make_closure(loss) if optimizer_name == 'alig' or optimizer_name.startswith('emo') else None

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(
            iterations=iterations,
            create_graph=should_use_create_graph(optimizer_name),
            use_amp=dtype == torch.bfloat16,
            closure_fn=closure_fn,
            threshold=1.4 if dtype != torch.complex64 and optimizer_name in ('spectralsphere', 'orthograd') else 1.5,
        )

    @pytest.mark.parametrize('optimizer_config', COMPILE_SUPPORTED_OPTIMIZERS, ids=ids)
    @pytest.mark.parametrize('foreach', [False, True])
    def test_tensor_lr(self, optimizer_config, foreach, environment):
        optimizer_name, config, iterations = optimizer_config
        config = config.copy()

        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)
        config['lr'] = torch.tensor(config['lr'], device=x_data.device)

        if optimizer_name == 'adamw' and foreach:
            if x_data.device.type == 'cpu':
                pytest.skip('Native AdamW foreach with a tensor learning rate requires a capturable device')
            config['capturable'] = True

        optimizer = build_optimizer(optimizer_name, model.parameters(), **config, foreach=foreach)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=iterations)


class TestOptimizerInterface:
    @pytest.mark.parametrize(
        'optimizer_name',
        [name for name in RECIPE_OPTIMIZER_NAMES if name not in ('lookahead', 'orthograd', 'schedulefree')],
    )
    def test_init_group(self, optimizer_name):
        parameters = TrainingModel() if optimizer_name in MODEL_OPTIMIZERS else [make_parameter()]
        optimizer = build_optimizer(optimizer_name, parameters)
        optimizer.init_group({'params': [], 'betas': (0.0, 0.0)})

    @pytest.mark.parametrize('optimizer_name', RECIPE_OPTIMIZER_NAMES)
    def test_closure(self, optimizer_name):
        if optimizer_name in MODEL_OPTIMIZERS:
            parameters = TrainingModel()
        else:
            param = torch.tensor([1.0, 0.0], requires_grad=True) if optimizer_name == 'orthograd' else make_parameter()
            param.grad = None
            parameters = [param]

        optimizer = build_optimizer(optimizer_name, parameters)
        optimizer.zero_grad()
        if optimizer_name == 'schedulefree':
            optimizer.train()

        if optimizer_name in ('ranger21', 'adai', 'adams'):
            with pytest.raises(ZeroParameterSizeError):
                optimizer.step(closure=dummy_closure)
        elif optimizer_name == 'alig':
            with pytest.raises(NoClosureError):
                optimizer.step()
        elif optimizer_name == 'orthograd':

            def closure():
                loss = param.sum()
                loss.backward()
                return loss

            assert optimizer.step(closure).item() == 1.0
            torch.testing.assert_close(param.grad, torch.tensor([0.0, 2.0**0.5]))
        else:
            optimizer.step(closure=dummy_closure)
