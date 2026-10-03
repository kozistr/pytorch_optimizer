import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.flash_adamw import compute_ecc_bits, reconstruct_fp32_param
from tests.fixtures import TrainingModel
from tests.utils import build_optimizer


class TestFlashAdamw:
    def test_flash_adamw_quantized_state_and_compressed_state_dict(self):
        param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        param.grad = torch.tensor([0.2, -0.3, 0.4])

        optimizer = build_optimizer('flashadamw', [param], lr=1e-2, weight_decay=0.0)
        optimizer.step()

        state = optimizer.state[param]
        assert state['exp_avg::quantized'].dtype == torch.int8
        assert state['exp_avg_sq::quantized'].dtype == torch.uint8
        assert state['exp_avg::scales'].dtype == torch.float16

        assert 'exp_avg' not in state
        assert optimizer.state_dict()['state'][0]['exp_avg::quantized'].dtype == torch.int8

        new_param = nn.Parameter(param.detach().clone())
        new_optimizer = build_optimizer('flashadamw', [new_param], lr=1e-2, weight_decay=0.0)
        new_optimizer.load_state_dict(optimizer.state_dict())
        assert new_optimizer.state[new_param]['exp_avg::quantized'].dtype == torch.int8

        new_param.grad = torch.tensor([-0.1, 0.5, -0.2])
        new_optimizer.step()

        assert torch.isfinite(new_param).all()

    def test_flash_adamw_empty_quantized_state(self):
        param = nn.Parameter(torch.empty(0))
        param.grad = torch.empty(0)

        optimizer = build_optimizer('flashadamw', [param], lr=1e-2, weight_decay=0.0)
        optimizer.step()

        param.grad = torch.empty(0)
        optimizer.step()

        state = optimizer.state[param]
        assert state['exp_avg::quantized'].numel() == 0
        assert state['exp_avg::scales'].numel() == 0

    def test_flash_adamw_loads_compressed_state_as_uncompressed_state(self):
        param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        param.grad = torch.tensor([0.2, -0.3, 0.4])

        optimizer = build_optimizer('flashadamw', [param], lr=1e-2, weight_decay=0.0)
        optimizer.step()

        new_param = nn.Parameter(param.detach().clone())
        state_dict = optimizer.state_dict()
        state_dict['param_groups'][0]['quantize'] = False

        new_optimizer = build_optimizer('flashadamw', [new_param], lr=1e-2, weight_decay=0.0, quantize=False)
        new_optimizer.load_state_dict(state_dict)
        new_state = new_optimizer.state[new_param]
        assert 'exp_avg' in new_state
        assert 'exp_avg::quantized' not in new_state

        uncompressed_optimizer = build_optimizer('flashadamw', [nn.Parameter(param.detach().clone())], quantize=False)
        uncompressed_optimizer.load_state_dict(uncompressed_optimizer.state_dict())
        assert list(uncompressed_optimizer.state_dict()['state'].values()) == [{}]

        raw_param = nn.Parameter(param.detach().clone())
        raw_param.grad = torch.zeros_like(raw_param)
        raw_optimizer = build_optimizer('flashadamw', [raw_param], quantize=False, compress_state_dict=False)
        raw_optimizer.step()
        assert 'exp_avg' in raw_optimizer.state_dict()['state'][0]

    def test_flash_adamw_uncompressed_state_dict_reloads_as_quantized_state(self):
        param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        param.grad = torch.tensor([0.2, -0.3, 0.4])

        optimizer = build_optimizer('flashadamw', [param], lr=1e-2, weight_decay=0.0, compress_state_dict=False)
        optimizer.step()

        state_dict = optimizer.state_dict()
        saved_state = state_dict['state'][0]
        assert 'exp_avg' in saved_state
        assert 'exp_avg::quantized' not in saved_state

        new_optimizer = build_optimizer(
            'flashadamw',
            [nn.Parameter(param.detach().clone())],
            lr=1e-2,
            weight_decay=0.0,
        )
        new_optimizer.load_state_dict(state_dict)
        new_state = next(iter(new_optimizer.state.values()))
        assert 'exp_avg::quantized' in new_state
        assert 'exp_avg' not in new_state

    @pytest.mark.parametrize(('master_weight_bits', 'error_dtype'), [(24, torch.int8), (32, torch.int16)])
    def test_flash_adamw_master_weight_bits(self, master_weight_bits, error_dtype):
        model = TrainingModel(dtype=torch.bfloat16)
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        optimizer = build_optimizer(
            'flashadamw',
            model.parameters(),
            lr=1e-2,
            weight_decay=0.0,
            quantize=False,
            master_weight_bits=master_weight_bits,
        )
        optimizer.step()

        fp32_state = optimizer.get_fp32_model_state_dict(model)
        assert all(tensor.dtype == torch.float32 for tensor in fp32_state.values())
        assert all(state['error_bits'].dtype == error_dtype for state in optimizer.state.values())

        updated = {name: tensor.add(0.01) for name, tensor in fp32_state.items()}
        optimizer.set_fp32_model_state_dict(model, updated)

        restored = optimizer.get_fp32_model_state_dict(model)
        assert all(torch.allclose(restored[name], updated[name], atol=1e-2) for name in updated)

    def test_flash_adamw_fresh_fp32_model_state_dict(self):
        model = TrainingModel(dtype=torch.bfloat16)

        optimizer = build_optimizer(
            'flashadamw',
            model.parameters(),
            lr=1e-2,
            weight_decay=0.0,
            quantize=False,
            master_weight_bits=24,
        )

        initial = optimizer.get_fp32_model_state_dict(model)
        assert all(tensor.dtype == torch.float32 for tensor in initial.values())

        optimizer.set_fp32_model_state_dict(model, {'missing.weight': torch.ones(1)})
        updated = {name: tensor.add(0.01) for name, tensor in initial.items()}

        optimizer.set_fp32_model_state_dict(model, updated)
        assert all('error_bits' in state for state in optimizer.state.values())

    def test_flash_adamw_ecc_helpers(self):
        fp32_param = torch.tensor([1.01, -2.02], dtype=torch.float32)
        narrow_param = fp32_param.to(torch.bfloat16)
        error_bits = compute_ecc_bits(fp32_param, narrow_param, master_byte_width=3)
        reconstructed = reconstruct_fp32_param(narrow_param, error_bits)

        assert error_bits.dtype == torch.int8
        assert torch.allclose(reconstructed, fp32_param, atol=1e-2)

        with pytest.raises(ValueError):
            compute_ecc_bits(fp32_param.to(torch.float16), narrow_param, master_byte_width=3)
        with pytest.raises(ValueError):
            compute_ecc_bits(fp32_param, fp32_param, master_byte_width=3)
        with pytest.raises(ValueError):
            compute_ecc_bits(fp32_param, narrow_param, master_byte_width=2)

        with pytest.raises(ValueError):
            reconstruct_fp32_param(fp32_param, error_bits)
        with pytest.raises(ValueError):
            reconstruct_fp32_param(narrow_param, torch.ones_like(narrow_param))

    def test_flash_adamw_numerics_guard_and_stats(self):
        param = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        optimizer = build_optimizer(
            'flashadamw', [param], lr=1e-3, weight_decay=0.0, quantize=False, check_numerics=True
        )

        optimizer.recompute_param_stats()
        optimizer.maybe_check_numerics(param, lr=0.0, master_byte_width=0)

        param.data.zero_()
        optimizer.param_absmax.pop(id(param), None)
        optimizer.maybe_check_numerics(param, lr=1e-3, master_byte_width=0)

        param.data.fill_(1.0)
        optimizer.param_absmax[id(param)] = float('nan')
        optimizer.maybe_check_numerics(param, lr=1e-3, master_byte_width=0)

        optimizer.param_absmax[id(param)] = 1.0
        with pytest.raises(ArithmeticError):
            optimizer.maybe_check_numerics(param, lr=1e-12, master_byte_width=2)

        empty_param = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
        empty_optimizer = build_optimizer('flashadamw', [empty_param], lr=1e-3, weight_decay=0.0, quantize=False)
        empty_optimizer.recompute_param_stats()
        assert empty_optimizer.param_absmax[id(empty_param)] == 0.0

    def test_flash_adamw_parameters(self):
        with pytest.raises(ValueError):
            load_optimizer('flashadamw')(None, master_weight_bits=16)

        with pytest.raises(NotImplementedError):
            load_optimizer('flashadamw')(None, fused=True)

        with pytest.raises(ValueError):
            load_optimizer('flashadamw')([nn.Parameter(torch.ones(1))], master_weight_bits=24)
