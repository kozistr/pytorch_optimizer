import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.utils.matrix import zero_power_via_newton_schulz_5
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestMuon:
    @pytest.mark.parametrize('optimizer_name', ['Muon', 'MuonClip', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_high_dimensions(self, optimizer_name):
        matrices = [make_parameter(shape, grad=1.0) for shape in ((1, 1, 1), (1, 1, 2, 2), (4, 1))]
        params = [
            {'params': [*matrices, make_parameter(grad=None)], 'use_muon': True},
            {'params': [make_parameter((1,), grad=None)], 'use_muon': False},
        ]
        optimizer = build_optimizer(optimizer_name, params, cautious=optimizer_name in ('Muon', 'AdaGO'))
        optimizer.step()
        assert all(torch.isfinite(param).all() for param in matrices)

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'MuonClip', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_use_muon_param(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)([{'params': [make_parameter()]}])

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'MuonClip', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_invalid_ns_coeffs(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)(
                [{'params': [make_parameter((2, 2))], 'use_muon': True}], ns_coeffs='invalid'
            )

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'MuonClip'])
    @pytest.mark.parametrize('group_option', [False, True])
    def test_invalid_ns_method(self, optimizer_name, group_option):
        options = {'ns_method': 'invalid'}
        group = {'params': [make_parameter()], 'use_muon': True, **(options if group_option else {})}
        with pytest.raises(ValueError, match='ns_method'):
            load_optimizer(optimizer_name)([group], **({} if group_option else options))


class TestMuonClip:
    @pytest.mark.parametrize('foreach', [False, True])
    @pytest.mark.parametrize('ns_method', ['newton_schulz', 'gram_newton_schulz'])
    def test_update_and_clip(self, foreach, ns_method, device):
        params = [make_parameter(shape, device=device) for shape in ((2, 4), (4, 2))]
        with torch.no_grad():
            for p in params:
                p.copy_(torch.arange(1, 9).reshape(p.shape) / 10.0)
        expected = [p.detach().clone() for p in params]
        buffers = [torch.zeros_like(p) for p in params]
        optimizer = build_optimizer(
            'muonclip',
            [{'params': params, 'use_muon': True, 'qk_clip': 'query', 'attention_name': 'attn'}],
            lr=0.1,
            momentum=0.5,
            weight_decay=2.0,
            qk_clip_alpha=0.25,
            ns_method=ns_method,
            foreach=foreach,
        )

        for step in range(2):
            for p, weight, buffer in zip(params, expected, buffers):
                grad = (torch.arange(8, device=device).reshape(p.shape) - 3.0).roll(step, dims=0) / 10.0
                p.grad = grad.clone()
                buffer.mul_(0.5).add_(grad)
                update = zero_power_via_newton_schulz_5(buffer, dtype=torch.float64).float()
                weight.mul_(1.0 - 0.1 * 2.0).add_(update, alpha=-0.1 * 0.2 * 2.0)
                weight[: p.size(0) // 2].mul_(0.25**0.25)

            optimizer.step(attention_max_logits={'attn': torch.tensor([400.0, 100.0])})

            for p, weight, buffer in zip(params, expected, buffers):
                torch.testing.assert_close(p, weight, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(optimizer.state[p]['momentum_buffer'], buffer)

    @pytest.mark.parametrize('shape', [(10, 3), (5, 2, 3)])
    def test_clip_projection_roles(self, shape):
        roles = ['query', 'key', 'query_rotary', 'key_rotary']
        params = [make_parameter(shape, grad=None) for _ in roles]
        groups = [
            {'params': [p], 'use_muon': True, 'qk_clip': role, 'attention_name': 'attn', 'qk_clip_threshold': 100.0}
            for p, role in zip(params, roles)
        ]
        omitted = make_parameter(shape, grad=None)
        groups.extend([
            {'params': [omitted], 'use_muon': True, 'qk_clip': 'query', 'attention_name': 'omitted'},
            {'params': [make_parameter(grad=None)], 'use_muon': True},
        ])
        with torch.no_grad():
            for p in [*params, omitted]:
                p.fill_(2.0)

        optimizer = build_optimizer('muonclip', groups, qk_clip_threshold=200.0)
        logits = torch.tensor([-200.0, 0.0, 100.0, 400.0, 1600.0], requires_grad=True)
        optimizer.step(attention_max_logits={'attn': logits})

        gamma = torch.tensor([1.0, 1.0, 1.0, 0.25, 0.0625])
        for p, exponent in zip(params, [0.5, 0.5, 1.0, 0.0]):
            expected = (2.0 * gamma.pow(exponent)).view(5, 1, 1).expand(5, 2, 3).reshape(shape)
            torch.testing.assert_close(p, expected)
        torch.testing.assert_close(omitted, torch.full_like(omitted, 2.0))
        torch.testing.assert_close(logits, torch.tensor([-200.0, 0.0, 100.0, 400.0, 1600.0]))

    @pytest.mark.parametrize(
        ('group_options', 'options', 'match'),
        [
            ({}, {'qk_clip_threshold': 0.0}, 'qk_clip_threshold'),
            ({}, {'qk_clip_alpha': -0.1}, 'qk_clip_alpha'),
            ({'qk_clip_threshold': 0.0}, {}, 'qk_clip_threshold'),
            ({'qk_clip_alpha': 1.1}, {}, 'qk_clip_alpha'),
            ({'qk_clip': 'invalid'}, {}, 'qk_clip'),
            ({'qk_clip': 'query'}, {}, 'attention_name'),
        ],
    )
    def test_clip_validation(self, group_options, options, match):
        group = {'params': [make_parameter()], 'use_muon': True, **group_options}
        with pytest.raises(ValueError, match=match):
            load_optimizer('muonclip')([group], **options)

    @pytest.mark.parametrize(
        ('shape', 'logits', 'match'),
        [
            ((2, 2), [], 'one-dimensional'),
            ((2, 2), [[100.0]], 'one-dimensional'),
            ((3, 2), [100.0, 200.0], 'divisible'),
            ((), [100.0], 'divisible'),
        ],
    )
    def test_clip_shape_validation(self, shape, logits, match):
        optimizer = build_optimizer('muonclip', [{
            'params': [make_parameter(shape, grad=None)],
            'use_muon': True,
            'qk_clip': 'query',
            'attention_name': 'attn',
        }])
        with pytest.raises(ValueError, match=match):
            optimizer.step(attention_max_logits={'attn': torch.tensor(logits)})
