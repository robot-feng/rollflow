from types import SimpleNamespace

import pytest
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead as StandardFlowmatchingActionHead,
)
from starVLA.model.modules.action_model.GR00T_RollFlow_ActionHeader import (
    RollFlowActionHead,
    RollFlowActionHeadConfig,
)
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.action_encoder import (
    ActionEncoder,
)
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.cross_attention_dit import DiT


def _config(**overrides):
    values = {
        "action_model_type": "DiT-B",
        "action_horizon": 4,
        "execution_horizon": 2,
        "action_dim": 3,
        "state_dim": 4,
        "hidden_size": 16,
        "add_pos_embed": True,
        "max_seq_len": 8,
        "num_target_vision_tokens": 2,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_inference_timesteps": 2,
        "num_timestep_buckets": 1000,
        "finite_difference_delta": 0.01,
        "p_k1": 0.0,
        "p_fm": 0.0,
        "fm_curriculum_steps": 0,
        "inference_steps": 2,
        "w_fm": 1.0,
        "w_lsd": 0.5,
        "use_ot": False,
        "clip_velocity": 0.0,
        "iterative_cold_start": False,
        "reset_cache_each_step": False,
        "diffusion_model_cfg": {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "output_dim": 16,
            "num_layers": 2,
            "cross_attention_dim": 12,
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
        },
    }
    values.update(overrides)
    return SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(**values)))


def test_action_encoder_accepts_tokenwise_source_time():
    torch.manual_seed(0)
    encoder = ActionEncoder(action_dim=3, hidden_size=16)
    actions = torch.zeros(2, 4, 3)
    source_time = torch.tensor([[[0.0], [0.0], [500.0], [500.0]]] * 2)

    encoded = encoder(actions, source_time)

    assert encoded.shape == (2, 4, 16)
    assert not torch.allclose(encoded[:, 0], encoded[:, 2])
    with pytest.raises(ValueError, match="timesteps must have shape"):
        encoder(actions, torch.zeros(2, 3))


def test_lsd_weight_warmup_reaches_core_config():
    model = RollFlowActionHead(_config(lsd_warmup_steps=250,finite_difference_delta_jitter=.5))
    assert model.rollflow.cfg.lsd_warmup_steps == 250
    assert model.rollflow.cfg.finite_difference_delta_jitter == .5
    assert RollFlowActionHeadConfig().lsd_warmup_steps == 0


def test_optional_clock_monitor_preserves_loss_gradients_and_rng():
    import copy

    torch.manual_seed(13)
    head = RollFlowActionHead(_config(w_lsd=0., clock_monitor_interval=20)).train()
    reference = copy.deepcopy(head)
    reference.clock_monitor_interval = 0
    context, actions, state = torch.randn(2,3,12), torch.randn(2,4,3), torch.randn(2,1,4)
    results = []
    for model in (head, reference):
        torch.manual_seed(812)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if model.clock_monitor_interval:
                # A no-grad probe must not poison a subsequent forward in the
                # same enclosing autocast context.
                model._clock_diagnostics()
            loss = model(context, actions, state, training_step=19)
        rng_after = torch.get_rng_state()
        loss.backward()
        assert model.model.transformer_blocks[0].attn1.to_q.weight.grad is not None
        results.append((loss.detach(), rng_after, [p.grad for p in model.parameters()]))
    assert head.last_loss_stats['clock_embedding_rms'] > 0
    assert head.last_loss_stats['clock_adaln_scale_max'] > 0
    assert 'clock_embedding_rms' not in reference.last_loss_stats
    torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
    assert torch.equal(results[0][1],results[1][1])
    for actual, expected in zip(results[0][2],results[1][2]):
        if actual is None or expected is None:
            assert actual is None and expected is None
        else:
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_dit_supports_endpoint_and_interval_per_token():
    torch.manual_seed(0)
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=16,
        num_layers=1,
        cross_attention_dim=12,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        norm_type="ada_norm",
    ).eval()
    hidden = torch.randn(2, 6, 16)
    context = torch.randn(2, 5, 12)
    target = torch.tensor([[1000.0] * 3 + [500.0] * 3] * 2)
    source = torch.tensor([[500.0] * 3 + [0.0] * 3] * 2)

    output = model(hidden, context, timestep=target, start_timestep=source)
    different_interval = model(hidden, context, timestep=target, start_timestep=target)
    legacy_output = model(hidden, context, timestep=torch.tensor([1000.0, 500.0]))

    assert output.shape == hidden.shape
    assert legacy_output.shape == hidden.shape
    assert not torch.allclose(output, different_interval)

    checkpoint_calls = []

    def checkpoint(module, *args):
        checkpoint_calls.append(module)
        return module(*args)

    model.enable_gradient_checkpointing(checkpoint)
    model.train()
    model(hidden, context, timestep=target, start_timestep=source)
    assert len(checkpoint_calls) == len(model.transformer_blocks)


def test_lsd_bf16_autocast_with_gradient_checkpointing_preserves_student_gradients():
    import copy
    from functools import partial
    from torch.utils.checkpoint import checkpoint
    from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import central_difference_lsd

    torch.manual_seed(98)
    models = [RollFlowActionHead(_config()).train()]
    models.append(copy.deepcopy(models[0]))
    models[1].model.enable_gradient_checkpointing(partial(checkpoint, use_reentrant=False))
    x0, x1 = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    s, t = torch.full((2, 4, 1), .2), torch.full((2, 4, 1), .7)
    context, state = torch.randn(2, 3, 12), torch.randn(2, 1, 4)
    outputs = []
    for head in models:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _ = central_difference_lsd(head._predict_velocity, x0, x1, s, t,
                delta=.01, context=context, model_kwargs={'state_features': head._encode_state(state)})
        loss.backward()
        gradients = dict(head.named_parameters())
        assert gradients['model.transformer_blocks.0.attn1.to_q.weight'].grad is not None
        outputs.append((loss.detach(), {n:p.grad for n,p in head.named_parameters()}))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    for name in outputs[0][1]:
        a, b = outputs[0][1][name], outputs[1][1][name]
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, msg=name)


def test_dit_can_suppress_zero_interval_embedding_gradients():
    torch.manual_seed(2)
    models = [
        DiT(
            num_attention_heads=2,
            attention_head_dim=8,
            output_dim=16,
            num_layers=1,
            cross_attention_dim=12,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
            norm_type="ada_norm",
            skip_zero_interval_embedding=skip,
        )
        for skip in (False, True)
    ]
    models[1].load_state_dict(models[0].state_dict())
    hidden = torch.randn(1, 5, 16)
    context = torch.randn(1, 3, 12)
    time = torch.full((1, 5), 500.0)
    outputs, interval_grads = [], []
    for model in models:
        output = model(
            hidden,
            context,
            timestep=time,
            start_timestep=time,
        )
        output.square().mean().backward()
        grads = [
            parameter.grad
            for parameter in model.interval_timestep_encoder.parameters()
            if parameter.grad is not None
        ]
        outputs.append(output.detach())
        interval_grads.append(grads)

    assert any(grad.abs().sum() > 0 for grad in interval_grads[0])
    assert all(torch.count_nonzero(grad) == 0 for grad in interval_grads[1])
    assert not torch.allclose(outputs[0], outputs[1])


@pytest.mark.parametrize("loss_space", ["velocity", "displacement"])
def test_rollflow_head_loss_backward_and_rolling_inference(loss_space):
    torch.manual_seed(0)
    model = RollFlowActionHead(_config(lsd_loss_space=loss_space))
    assert model.rollflow.cfg.lsd_loss_space == loss_space
    context = torch.randn(2, 5, 12)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 1, 4)
    context_mask = torch.tensor([[True, True, True, True, False]] * 2)

    times = model.rollflow.time.deployment(
        2,
        refinement_steps=2,
        cold=False,
        device="cpu",
    )
    velocity = model._predict_velocity(
        actions,
        times.s,
        times.t,
        context,
        state_features=model._encode_state(state),
        encoder_attention_mask=context_mask,
    )
    assert velocity.shape == actions.shape

    loss = model(context, actions, state, encoder_attention_mask=context_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    endpoint_grad = model.model.timestep_encoder.timestep_embedder.linear_1.weight.grad
    interval_grad = model.model.interval_timestep_encoder.timestep_embedder.linear_1.weight.grad
    assert endpoint_grad is not None and torch.isfinite(endpoint_grad).all()
    assert interval_grad is not None and torch.isfinite(interval_grad).all()
    assert endpoint_grad.abs().sum() > 0
    assert interval_grad.abs().sum() > 0
    assert model.last_loss_stats["num_time_groups"] == 2

    model.eval()
    cold_chunk = model.predict_action(context, state, encoder_attention_mask=context_mask)
    warm_chunk = model.predict_action(context, state, encoder_attention_mask=context_mask)
    assert cold_chunk.shape == warm_chunk.shape == (2, 2, 3)
    assert model.cache_info.shape == (2, 4, 3)

    model.reset_cache()
    assert model.cache_info is None

    reloaded = RollFlowActionHead(_config())
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert "model.interval_timestep_encoder.timestep_embedder.linear_1.weight" in model.state_dict()


def test_rollflow_padding_mask_blocks_padded_action_keys():
    torch.manual_seed(1)
    model = RollFlowActionHead(_config()).eval()
    context = torch.randn(2, 5, 12)
    actions = torch.randn(2, 4, 3)
    pad = torch.tensor([[False, False, True, True], [False, False, False, True]])
    changed = actions.clone()
    changed[pad] = torch.randn_like(changed[pad]) * 100
    source_time = torch.full((2, 4), 0.25)
    target_time = torch.full((2, 4), 0.75)

    masked_a = model._predict_velocity(
        actions, source_time, target_time, context, action_padding_mask=pad
    )
    masked_b = model._predict_velocity(
        changed, source_time, target_time, context, action_padding_mask=pad
    )
    torch.testing.assert_close(masked_a[~pad], masked_b[~pad])

    unmasked_a = model._predict_velocity(actions, source_time, target_time, context)
    unmasked_b = model._predict_velocity(changed, source_time, target_time, context)
    assert not torch.allclose(unmasked_a[~pad], unmasked_b[~pad])

    model.train()
    loss = model(context, actions, action_padding_mask=pad)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_rollflow_head_disables_stochastic_finite_differences():
    diffusion_cfg = _config().framework.action_model.diffusion_model_cfg.copy()
    diffusion_cfg["dropout"] = 0.1
    model = RollFlowActionHead(_config(diffusion_model_cfg=diffusion_cfg))
    assert model.model.config.dropout == 0.0
    assert model.model.config.final_dropout is False


def test_rollflow_head_uses_trainer_max_steps_for_fm_schedule():
    config = _config()
    config.trainer = SimpleNamespace(max_train_steps=1234)
    config.framework.action_model.fm_only_steps = 321

    model = RollFlowActionHead(config)

    assert model.rollflow.cfg.total_train_steps == 1234
    assert model.rollflow.cfg.fm_curriculum_steps == 321


def test_config_defaults_and_legacy_gr00t_checkpoint_loading():
    defaults = RollFlowActionHeadConfig()
    assert defaults.diffusion_model_cfg == {}
    assert defaults.to_dict()["finite_difference_delta"] == 0.01
    assert defaults.to_dict()["use_ot"] is True
    assert defaults.to_dict()["p_k1"] == 0.7
    assert defaults.to_dict()["p_fm"] == 0.3
    assert defaults.to_dict()["fm_curriculum_steps"] == 5000

    diffusion_cfg = _config().framework.action_model.diffusion_model_cfg.copy()
    diffusion_cfg.update(num_attention_heads=12, attention_head_dim=64, num_layers=1)
    config = _config(diffusion_model_cfg=diffusion_cfg)
    standard = StandardFlowmatchingActionHead(config)
    rollflow = RollFlowActionHead(config)
    rollflow.load_state_dict(standard.state_dict(), strict=True)

    endpoint = rollflow.model.timestep_encoder.state_dict()
    interval = rollflow.model.interval_timestep_encoder.state_dict()
    assert endpoint.keys() == interval.keys()
    for name in endpoint:
        torch.testing.assert_close(endpoint[name], interval[name])
