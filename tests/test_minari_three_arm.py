"""Contracts for the Minari three-arm isolation experiment."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "minari_three_arm", Path(__file__).resolve().parents[1] / "examples/simBenchmarks/Minari/run.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def cfg(variant):
    args = SimpleNamespace(variant=variant, horizon=8, chunk=2, width=16, nfe=4, repeat=4,
        layers=2, p_k1=0., p_fm=.3, fm_warmup=5, w_lsd=.05, loss_space='velocity',
        lsd_ramp=2, delta=.01, ot=False, skip_zero_interval=False)
    return m.build_config(args, 3, 2)


def test_actual_heads_shared_initialization_and_matched_fm_warmup():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = m.make_model(cfg('fm'), 20, 'cpu', 'fm')
    left = m.make_model(cfg('rolling_fm'), 20, 'cpu', 'rolling_fm')
    right = m.make_model(cfg('rolling_lsd'), 20, 'cpu', 'rolling_lsd')
    initial = copy.deepcopy(base.action_model.state_dict())
    left.action_model.load_state_dict(initial, strict=True)
    right.action_model.load_state_dict(initial, strict=True)
    assert isinstance(base.action_model, m.standard.FlowmatchingActionHead)
    assert type(left.action_model) is type(right.action_model) is m.rolling.RollFlowActionHead
    for name, value in initial.items():
        torch.testing.assert_close(left.action_model.state_dict()[name], value, rtol=0, atol=0)
    state = torch.randn(4, 3)
    actions = torch.randn(4, 8, 2)
    losses = []
    for model in (left, right):
        torch.manual_seed(123)
        losses.append(model.loss(state, actions, training_step=1))
        losses[-1].backward()
        assert model.last_loss_stats['num_time_groups'] > 1
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for p, q in zip(left.parameters(), right.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    torch.manual_seed(123)
    loss = right.loss(state, actions, training_step=20)
    assert torch.isfinite(loss)
    assert right.last_loss_stats['active_lsd_frac'] > 0
    assert right.last_loss_stats['weighted_lsd_loss'] > 0


@pytest.mark.parametrize('variant', ['fm', 'rolling_fm', 'rolling_lsd'])
def test_chunk_reset_reload_and_framework_api(variant):
    model = m.make_model(cfg(variant), 20, 'cpu', variant).eval()
    clone = m.make_model(cfg(variant), 20, 'cpu', variant).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    rng_before=torch.get_rng_state()
    diagnostics=m.conditioning_metrics(model)
    assert all(np.isfinite(value) and value>=0 for value in diagnostics.values())
    assert torch.equal(rng_before,torch.get_rng_state())
    examples = [{'state':np.ones(3, np.float32), 'action':np.ones((8,2), np.float32)}]
    assert torch.isfinite(model(examples, training_step=20)['action_loss'])
    outputs=[]
    for instance in (model, clone):
        instance.reset();torch.manual_seed(45)
        chunks=[instance.predict_action(examples)['normalized_actions'] for _ in range(3)]
        assert all(chunk.shape==(1,2,2) for chunk in chunks)
        outputs.append(np.concatenate(chunks,1))
    np.testing.assert_array_equal(*outputs)


def test_complete_windows_and_padding_guard():
    data=[(0,np.zeros((13,3),np.float32),np.ones((12,2),np.float32))]
    stats=dict(state_mean=np.zeros(3),state_std=np.ones(3),action_center=np.zeros(2),action_scale=np.ones(2))
    _, actions, pad=m.sample(data,stats,8,32,np.random.default_rng(0),'cpu')
    assert not pad.any()
    assert actions.eq(1).all()
    with pytest.raises(ValueError,match='No complete'):
        m.sample(data,stats,32,2,np.random.default_rng(0),'cpu')
    model=m.make_model(cfg('fm'),20,'cpu','fm')
    with pytest.raises(ValueError,match='padding'):
        model.loss(torch.zeros(1,3),torch.zeros(1,8,2),pad=torch.ones(1,8,dtype=torch.bool))


@pytest.mark.parametrize('variant', ['fm', 'rolling_fm', 'rolling_lsd'])
def test_repeat_is_identical_to_manual_batch_replication(variant):
    config = cfg(variant)
    model = m.make_model(config, 20, 'cpu', variant)
    state, actions = torch.randn(3,3), torch.randn(3,8,2)
    pad = torch.zeros(3,8,dtype=torch.bool)
    torch.manual_seed(981)
    automatic = model.loss(state, actions, pad=pad, training_step=20)
    model.repeated_diffusion_steps = 1
    torch.manual_seed(981)
    manual = model.loss(state.repeat(4,1), actions.repeat(4,1,1),
                        pad=pad.repeat(4,1), training_step=20)
    torch.testing.assert_close(automatic, manual, rtol=0, atol=0)


def test_checkpoint_replay_preserves_episode_order():
    episodes=[(8,None,None),(13,None,None)]
    assert [ep[0] for ep in m.restore_test_order(episodes,[13,8])]==[13,8]
    with pytest.raises(ValueError,match='missing'):
        m.restore_test_order(episodes,[99])


def test_evaluation_refinements_and_legacy_checkpoint_defaults():
    args=SimpleNamespace(horizon=32,chunk=8,nfe=4)
    assert m.evaluation_refinements(args)==[1,4]
    args.eval_nfe=[4,1,2,2]
    assert m.evaluation_refinements(args)==[1,2,4]
    args.eval_nfe=[5]
    with pytest.raises(ValueError,match='refinement'):
        m.evaluation_refinements(args)


def test_dataset_fingerprint_includes_split_order_and_actions():
    episode=(7,np.zeros((5,3),np.float32),np.zeros((4,2),np.float32))
    data={'train':[episode],'validation':[episode],'test':[episode]}
    original=m.dataset_fingerprint(data)
    changed=copy.deepcopy(data)
    changed['test'][0][2][0,0]=1
    assert m.dataset_fingerprint(changed)!=original
    changed=copy.deepcopy(data)
    changed['test'][0]=(8,changed['test'][0][1],changed['test'][0][2])
    assert m.dataset_fingerprint(changed)!=original
