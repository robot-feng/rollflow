import math

import pytest
import torch
from torch import nn

from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    RollFlow,
    RollFlowConfig,
    StaircaseTimeSampler,
    central_difference_lsd,
    ot_match,
)


class _LinearPathOracle(nn.Module):
    """Exact average velocity from an arbitrary source point to context=x1."""

    def forward(self, z, source_time, target_time, context, **kwargs):
        del target_time, kwargs
        return (context - z) / (1.0 - source_time)


class _LearnableConstant(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))
        self.batch_sizes = []
        self.grad_enabled = []

    def forward(self, z, source_time, target_time, context, **kwargs):
        del source_time, target_time, context, kwargs
        self.batch_sizes.append(z.shape[0])
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.value.expand_as(z)


class _CaptureQuery(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z, source_time, target_time, context, **kwargs):
        del context, kwargs
        self.calls.append((source_time.detach().clone(), target_time.detach().clone()))
        return torch.zeros_like(z)


def _config(**overrides):
    values = {
        "horizon": 8,
        "action_dim": 1,
        "chunk_size": 1,
        "finite_difference_delta": 0.01,
        "inference_steps": 4,
        "p_k1": 1.0,
        "p_fm": 0.0,
        "fm_curriculum_steps": 0,
        "w_fm": 1.0,
        "w_lsd": 0.25,
        "use_ot": False,
        "iterative_cold_start": True,
    }
    values.update(overrides)
    return RollFlowConfig(**values)


def test_training_times_are_valid_for_every_divisor_k():
    sampler = StaircaseTimeSampler(_config())

    for k in (1, 2, 4, 8):
        times = sampler.sample_training(
            5,
            device="cpu",
            p_fm=0.0,
            num_time_groups=k,
        )
        assert torch.all(times.s >= 0)
        assert torch.all(times.s <= times.t)
        assert torch.all(times.t <= 1)
        assert torch.all(times.t[times.active] + 0.01 <= 1.0 + 1e-6)
        assert times.num_time_groups == k
        assert times.block_size == 8 // k
        assert times.ratio.shape == (5, 1, 1)
        group_width = times.block_size
        grouped_t = times.t[:, ::group_width, 0]
        assert torch.all(grouped_t[:, :-1] >= grouped_t[:, 1:])
        torch.testing.assert_close(times.s, times.ratio * times.t)


def test_grouped_training_times_match_32_by_8_design():
    cfg = RollFlowConfig(
        horizon=32,
        action_dim=1,
        chunk_size=8,
        finite_difference_delta=0.01,
        inference_steps=4,
    )
    sampler = StaircaseTimeSampler(cfg)
    for k, block_size in ((1, 4), (2, 2), (4, 1)):
        sampled = sampler.sample_training(
            3, device="cpu", p_fm=1.0, num_time_groups=k
        )
        assert sampled.block_size == block_size
        assert sampled.num_time_groups == k
        assert not sampled.active.any()
        torch.testing.assert_close(sampled.s, sampled.t)


def test_k_sampling_probabilities_and_config_validation():
    generator = torch.Generator().manual_seed(0)
    sampler = StaircaseTimeSampler(_config(p_k1=0.7))
    counts = {1: 0, 2: 0, 4: 0, 8: 0}
    for _ in range(4000):
        counts[sampler._sample_k(torch.device("cpu"), generator)] += 1
    assert counts[1] / 4000 == pytest.approx(0.7, abs=0.03)
    for k in (2, 4, 8):
        assert counts[k] / 4000 == pytest.approx(0.1, abs=0.025)
    with pytest.raises(ValueError, match="p_fm"):
        _config(p_fm=1.1)


def test_inference_velocity_mode_switches_query_without_changing_step_count():
    cfg = _config(iterative_cold_start=False, reset_cache_each_step=True)
    flow = RollFlow(cfg)
    model = _CaptureQuery()

    flow.step(model, batch=1, velocity_mode="average")
    average_s, average_t = model.calls[-1]
    assert torch.any(average_t > average_s)

    flow.step(model, batch=1, velocity_mode="instant")
    instant_s, instant_t = model.calls[-1]
    torch.testing.assert_close(instant_s, instant_t)

    with pytest.raises(ValueError, match="velocity_mode"):
        flow.step(model, batch=1, velocity_mode="bad")


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("iterative_cold_start", [False, True])
def test_instant_rolling_queries_diagonal_and_advances_real_interval(k, iterative_cold_start):
    class UnitVelocity(_CaptureQuery):
        def forward(self, z, source_time, target_time, context, **kwargs):
            super().forward(z, source_time, target_time, context, **kwargs)
            return torch.ones_like(z)

    cfg = _config(horizon=32, chunk_size=4, inference_steps=k,
                  iterative_cold_start=iterative_cold_start)
    flow, model = RollFlow(cfg), UnitVelocity()
    torch.manual_seed(123)
    noise = torch.randn(1, 32, 1)
    torch.manual_seed(123)
    cold_chunk = flow.step(model, batch=1, velocity_mode="instant")
    assert len(model.calls) == (k if iterative_cold_start else 1)
    assert all(torch.equal(source, target) for source, target in model.calls)
    torch.testing.assert_close(cold_chunk, noise[:, :4] + 1.0)

    shifted_front = flow._cache[:, 4:8].clone()
    previous_calls = len(model.calls)
    warm_chunk = flow.step(model, batch=1, velocity_mode="instant")
    assert len(model.calls) == previous_calls + 1
    source, target = model.calls[-1]
    assert torch.equal(source, target)
    torch.testing.assert_close(warm_chunk, shifted_front + 1.0 / k)


def test_h128_k4_returns_one_32_action_execution_chunk():
    cfg = _config(
        horizon=128,
        chunk_size=32,
        inference_steps=4,
        iterative_cold_start=False,
        reset_cache_each_step=True,
    )
    flow = RollFlow(cfg)
    output = flow.step(_CaptureQuery(), batch=2)

    assert output.shape == (2, 32, 1)
    assert flow.cache_info is not None
    assert flow.cache_info.shape == (2, 128, 1)
    assert flow.cache_info.refinement_steps == 4


def test_fm_curriculum_starts_diagonal_and_reaches_target_probability():
    torch.manual_seed(0)
    rollflow = RollFlow(
        _config(p_fm=0.2, fm_curriculum_steps=100, total_train_steps=200)
    )
    model = _LearnableConstant()
    actions = torch.randn(32, 8, 1)

    _, start = rollflow.loss(model, actions, step=0)
    assert start["p_fm"] == 1.0
    assert start["fm_only_frac"] == 1.0
    assert start["active_lsd_frac"] == 0.0
    assert model.batch_sizes == [32]

    _, warmup = rollflow.loss(model, actions, step=50)
    _, boundary = rollflow.loss(model, actions, step=100)
    _, middle = rollflow.loss(model, actions, step=150)
    _, end = rollflow.loss(model, actions, step=200)
    assert warmup["p_fm"] == pytest.approx(1.0)
    assert boundary["p_fm"] == pytest.approx(1.0)
    assert middle["p_fm"] == pytest.approx(0.6)
    assert end["p_fm"] == pytest.approx(0.2)


def test_two_thirds_fm_schedule_enables_lsd_only_after_30000():
    torch.manual_seed(42)
    flow = RollFlow(_config(p_fm=0.3, fm_curriculum_steps=30000,
                            total_train_steps=45000, lsd_warmup_steps=1000))
    model = _LearnableConstant()
    actions = torch.randn(128, 8, 1)
    for step in (0, 29999, 30000):
        _, stats = flow.loss(model, actions, step=step)
        assert stats["p_fm"] == 1.0
        assert stats["fm_only_frac"] == 1.0
        assert stats["active_lsd_frac"] == 0.0
    _, first = flow.loss(model, actions, step=30001)
    assert first["p_fm"] < 1.0
    assert first["lsd_weight"] == pytest.approx(0.25 / 1000)
    _, middle = flow.loss(model, actions, step=37500)
    assert middle["p_fm"] == pytest.approx(0.65)
    assert middle["lsd_weight"] == pytest.approx(0.25)
    assert middle["active_lsd_frac"] > 0
    _, end = flow.loss(model, actions, step=45000)
    assert end["p_fm"] == pytest.approx(0.3)
    assert end["lsd_weight"] == pytest.approx(0.25)


@pytest.mark.parametrize("loss_space", ["velocity", "displacement"])
def test_linear_path_oracle_has_zero_fm_and_lsd_error(loss_space):
    torch.manual_seed(0)
    cfg = _config()
    x0 = torch.randn(3, cfg.horizon, cfg.action_dim)
    x1 = torch.randn_like(x0)
    times = StaircaseTimeSampler(cfg).sample_training(
        x0.shape[0],
        device=x0.device,
        p_fm=0.0,
        num_time_groups=4,
    )

    loss, stats = central_difference_lsd(
        _LinearPathOracle(),
        x0,
        x1,
        times.s,
        times.t,
        delta=cfg.finite_difference_delta,
        active=times.active,
        context=x1,
        w_fm=cfg.w_fm,
        w_lsd=cfg.w_lsd,
        lsd_loss_space=loss_space,
    )

    assert loss < 1e-8
    assert stats["fm_loss"] < 1e-10
    assert stats["lsd_loss"] < 1e-7


def test_loss_retains_gradients_only_for_tangent_and_local_paths():
    torch.manual_seed(0)
    rollflow = RollFlow(_config())
    model = _LearnableConstant()
    actions = torch.randn(3, 8, 1)

    loss, _ = rollflow.loss(model, actions)
    loss.backward()

    assert model.batch_sizes == [3, 3, 9]
    assert model.grad_enabled == [False, False, True]
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)


@pytest.mark.parametrize("loss_space", ["velocity", "displacement"])
def test_mixed_fm_batch_runs_lsd_only_for_active_rows(loss_space):
    model = _LearnableConstant()
    x0 = torch.randn(4, 8, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((4, 8, 1), 0.6)
    s = t.clone()
    s[[1, 3]] = 0.2

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        lsd_loss_space=loss_space,
    )
    loss.backward()

    assert model.batch_sizes == [2, 2, 8]
    assert model.grad_enabled == [False, False, True]
    assert stats["active_lsd_frac"] == 0.5


@pytest.mark.parametrize("loss_space", ["velocity", "displacement"])
def test_padded_rows_are_excluded_from_lsd_forwards(loss_space):
    model = _LearnableConstant()
    x0 = torch.randn(4, 8, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((4, 8, 1), 0.6)
    s = t.clone()
    s[[1, 3]] = 0.2
    pad = torch.zeros(4, 8, dtype=torch.bool)
    pad[3] = True

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        pad=pad,
        lsd_loss_space=loss_space,
    )
    loss.backward()

    assert model.batch_sizes == [1, 1, 6]
    # The diagnostic is normalized by valid tokens, excluding the all-padding row.
    assert stats["active_lsd_frac"] == pytest.approx(1.0 / 3.0)


def test_training_input_shapes_are_checked():
    x0 = torch.randn(2, 8, 1)
    x1 = torch.randn_like(x0)
    times = torch.zeros(2, 8, 1)
    with pytest.raises(ValueError, match="x0 and x1"):
        central_difference_lsd(
            _LearnableConstant(), x0, x1[:, :-1], times, times, delta=0.01
        )
    with pytest.raises(ValueError, match="s and t"):
        central_difference_lsd(
            _LearnableConstant(), x0, x1, times.squeeze(-1), times.squeeze(-1), delta=0.01
        )


def test_all_padding_returns_graph_connected_zero():
    torch.manual_seed(0)
    rollflow = RollFlow(_config())
    model = _LearnableConstant()
    actions = torch.randn(2, 8, 1)
    padding = torch.ones(2, 8, dtype=torch.bool)

    loss, _ = rollflow.loss(model, actions, pad=padding)
    loss.backward()

    assert loss.item() == 0.0
    assert model.value.grad is not None
    assert model.value.grad.item() == 0.0
    assert model.batch_sizes == [2]


def test_ot_matching_ignores_padded_target_values():
    x1 = torch.tensor([[[0.0], [1000.0]], [[10.0], [-1000.0]]])
    x0 = torch.tensor([[[0.0], [-1000.0]], [[10.0], [1000.0]]])
    pad = torch.tensor([[False, True], [False, True]])

    matched = ot_match(x1, x0, pad=pad)

    torch.testing.assert_close(matched, x0)


def test_oracle_rolls_sine_from_minus_20pi_to_plus_20pi():
    torch.manual_seed(0)
    cfg = _config()
    rollflow = RollFlow(cfg)
    oracle = _LinearPathOracle()
    phase_step = math.pi / 16.0
    phases = torch.linspace(-20.0 * math.pi, 20.0 * math.pi, 641)
    offsets = torch.arange(cfg.horizon) * phase_step
    predictions = []

    for phase in phases:
        target_chunk = torch.sin(phase + offsets)[None, :, None]
        action = rollflow.step(oracle, batch=1, context=target_chunk)
        predictions.append(action[0, 0, 0])

    torch.testing.assert_close(torch.stack(predictions), torch.sin(phases), atol=2e-5, rtol=2e-5)
    assert rollflow.cache_info is not None
    assert rollflow.cache_info.shape == (1, 8, 1)
    assert rollflow.cache_info.refinement_steps == 4

    rollflow.reset()
    assert rollflow.cache_info is None
def test_lsd_mask_and_validation_agree_at_float32_boundary():
    from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
        _lsd_mask, _validate_times,
    )

    s = torch.tensor([[[0.5490913987159729]]], dtype=torch.float32)
    t = torch.tensor([[[0.5590924024581909]]], dtype=torch.float32)
    active = _lsd_mask(s, t, 0.01)
    assert active.item()
    _validate_times(s, t, 0.01, active)
