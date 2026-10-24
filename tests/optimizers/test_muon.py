import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.muon import get_adjusted_lr
from pytorch_optimizer.optimizer.utils.matrix import zero_power_via_newton_schulz_5
from tests.fixtures import make_parameter
from tests.utils import build_optimizer
from tests.utils.distributed import run_distributed


def _check_sharded_muon(_rank, optimizer_name, foreach, ns_method):
    dtensor = pytest.importorskip('torch.distributed.tensor')
    mesh = torch.distributed.device_mesh.init_device_mesh('cpu', (2,))
    shapes = [(3, 2), (2, 3), (3, 2, 1), (6, 2), (1, 2), (3, 2), (3,)]
    reference = [make_parameter(shape, grad=None) for shape in shapes]
    with torch.no_grad():
        for p in reference:
            p.copy_(torch.arange(p.numel()).reshape(p.shape) / 10.0)
    sharded = [
        dtensor.distribute_tensor(p.detach(), mesh, [dtensor.Shard(1 if index == 5 else 0)]).requires_grad_()
        for index, p in enumerate(reference)
    ]

    def groups(params):
        return [
            {'params': params[:3], 'use_muon': True},
            {'params': [params[3], params[5]], 'use_muon': True, 'qk_clip': 'query', 'attention_name': 'attn'},
            {'params': [params[4]], 'use_muon': True, 'qk_clip': 'query', 'attention_name': 'one_head'},
            {'params': [params[6]], 'use_muon': False},
        ]

    options = {'foreach': foreach, 'ns_method': ns_method, 'weight_decay': 0.1}
    optimizer = build_optimizer(optimizer_name, groups(sharded), **options)
    expected_optimizer = build_optimizer(optimizer_name, groups(reference), **options)
    logits = {'attn': torch.tensor([400.0, 100.0, 1600.0]), 'one_head': torch.tensor([400.0])}
    step_options = {'attention_max_logits': logits} if optimizer_name == 'muonclip' else {}

    for step in range(2):
        for p, expected in zip(sharded, reference):
            grad = (torch.arange(p.numel()).reshape(p.shape) - 2.0).roll(step, dims=0) / 10.0
            p.grad = dtensor.distribute_tensor(grad, mesh, p.placements)
            expected.grad = grad.clone()
        optimizer.step(**step_options)
        expected_optimizer.step(**step_options)

        for p, expected in zip(sharded, reference):
            torch.testing.assert_close(p.full_tensor(), expected, atol=1e-7, rtol=1e-6)
            for key, state in optimizer.state[p].items():
                torch.testing.assert_close(
                    state.full_tensor(), expected_optimizer.state[expected][key].reshape(state.shape)
                )
                assert state.placements == p.placements
                assert state.to_local().shape == p.to_local().shape


class TestMuon:
    @pytest.mark.parametrize(
        ('shape', 'keller_factor', 'moonlight_factor'),
        [
            ((16, 2, 2), 2.0, 0.8),
            ((4, 16), 1.0, 0.8),
        ],
    )
    def test_adjusted_lr_factors(self, shape, keller_factor, moonlight_factor):
        torch.testing.assert_close(get_adjusted_lr(0.5, shape), 0.5 * keller_factor)
        torch.testing.assert_close(get_adjusted_lr(0.5, shape, use_adjusted_lr=True), 0.5 * moonlight_factor)

    @pytest.mark.parametrize('optimizer_name', ['muon', 'normuon'])
    def test_foreach_shape_scaling(self, optimizer_name):
        params = [make_parameter(shape) for shape in ((2, 4), (4, 2))]
        expected_norms = []
        for p, factor in zip(params, (1.0, 2.0**0.5)):
            p.grad = (torch.arange(8).reshape(p.shape) - 3.0) / 10.0
            update = zero_power_via_newton_schulz_5(p.grad.reshape(p.size(0), -1)).float()
            expected_norms.append(0.1 * factor * update.norm())

        optimizer = build_optimizer(
            optimizer_name,
            [{'params': params, 'use_muon': True}],
            lr=0.1,
            momentum=0.0,
            foreach=True,
        )
        optimizer.step()

        for p, expected_norm in zip(params, expected_norms):
            torch.testing.assert_close(p.norm(), expected_norm, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        ('optimizer_name', 'foreach', 'ns_method'),
        [
            ('muon', False, 'newton_schulz'),
            ('muon', True, 'gram_newton_schulz'),
            ('muonclip', False, 'gram_newton_schulz'),
            ('muonclip', True, 'newton_schulz'),
            ('adamuon', False, 'newton_schulz'),
            ('adamuon', True, 'newton_schulz'),
        ],
    )
    @pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='Gloo is unavailable')
    def test_sharded_updates(self, optimizer_name, foreach, ns_method, tmp_path):
        pytest.importorskip('torch.distributed.tensor')
        run_distributed(_check_sharded_muon, tmp_path / 'store', optimizer_name, foreach, ns_method)

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'MuonClip', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_high_dimensions(self, optimizer_name):
        matrices = [make_parameter(shape, grad=1.0) for shape in ((1, 1, 1), (1, 1, 2, 2), (4, 1))]
        params = [
            {'params': [*matrices, make_parameter(grad=None)], 'use_muon': True},
            {'params': [make_parameter((1,), grad=None)], 'use_muon': False},
        ]
        optimizer = build_optimizer(
            optimizer_name,
            params,
            cautious=optimizer_name in ('Muon', 'AdaGO'),
            foreach=optimizer_name not in ('Muon', 'AdaGO'),
        )
        optimizer.step()
        assert all(torch.isfinite(param).all() for param in matrices)

        if optimizer_name == 'AdaMuon':
            expected_rms = torch.tensor(0.2 * optimizer.param_groups[0]['lr'])
            for param in matrices:
                torch.testing.assert_close(param.square().mean().sqrt(), expected_rms, rtol=1e-2, atol=1e-5)

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
