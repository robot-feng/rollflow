"""Tests complementary portable kernels; not claims about production VLA weights."""
import torch
import torch.nn as nn
import pytest
from rolling_flow_torch import (
    FlowConvention, RollingFlow, sample_grouped_time, training_group_choices,
)
from lsd_flow_torch import lsd_loss, RollingLSD


def test_optional_instant_group_choices_differ_from_original():
    assert training_group_choices(64) == (1, 2, 4, 8, 16, 32, 64)
    assert training_group_choices(12) == (1, 2, 3, 4, 6, 12)


@pytest.mark.parametrize('groups',[1,2,4,8,16,32,64])
def test_optional_native_sampler_keeps_BH_and_sign(groups):
    calls=[]
    def sample(n):
        calls.append(n)
        return torch.linspace(.1,.9,n)
    for convention in (FlowConvention.NOISE_AT_ZERO,FlowConvention.NOISE_AT_ONE):
        out=sample_grouped_time(sample,batch_size=2,horizon=64,num_groups=groups,convention=convention)
        assert out.shape==(2,64)
        assert calls[-1]==2*groups
        if groups==1:
            assert torch.allclose(out[0], out[0,0].expand(64))
        else:
            delta=torch.diff(out,dim=1)
            assert (delta<=1e-7).all() if convention is FlowConvention.NOISE_AT_ZERO else (delta>=-1e-7).all()


class ConstantField(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.2))
        self.calls=[]
    def forward(self,x,t):
        self.calls.append((x.shape, t.clone()))
        return self.scale*torch.ones_like(x)


@pytest.mark.parametrize('conv',[FlowConvention.NOISE_AT_ZERO,FlowConvention.NOISE_AT_ONE])
def test_instant_rolling_consumption_and_sign(conv):
    state=RollingFlow(action_dim=3,horizon=64,execution_horizon=16,convention=conv)
    field=ConstantField()
    cold_noise=torch.zeros(1,64,3)
    out=state.step(field.forward,noise=cold_noise,stream_ids=(100,),context_key='episodeA')
    assert out.shape==(1,16,3)
    assert state.last_nfe==4
    assert len(field.calls)==4
    sign=1 if conv is FlowConvention.NOISE_AT_ZERO else -1
    assert torch.allclose(out,torch.full_like(out,.2*sign),atol=1e-6)
    with pytest.raises(ValueError):
        state.step(field.forward,noise=torch.zeros(1,16,3),consumed_actions=0,stream_ids=(100,),context_key='episodeA')
    prev=state._cache.clone()
    assert torch.equal(prev,state._cache) # no premature cache commit
    out2=state.step(field.forward,noise=torch.zeros(1,16,3),consumed_actions=16,stream_ids=(100,),context_key='episodeA')
    assert out2.shape==(1,16,3)
    assert state.last_nfe==1
    assert len(field.calls)==5
    with pytest.raises(ValueError):
        state.step(field.forward,noise=torch.zeros(1,16,3),consumed_actions=16,stream_ids=(101,),context_key='episodeA')


def test_reference_rolling_rejects_bad_active_update_without_cache_commit():
    state=RollingFlow(action_dim=2,horizon=8,execution_horizon=2,convention=FlowConvention.NOISE_AT_ZERO)
    state.step(lambda x,t: torch.zeros_like(x),noise=torch.zeros(1,8,2))
    before=state._cache.clone()
    with pytest.raises(FloatingPointError):
        state.step(lambda x,t: torch.full_like(x,float('nan')),noise=torch.zeros(1,2,2),consumed_actions=2)
    assert torch.equal(before,state._cache)


class TwoTimeField(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.2))
        self.calls=[]
    def forward(self,x,s,t):
        self.calls.append((torch.is_grad_enabled(),x.shape))
        return self.weight * (x + .05 * s.unsqueeze(-1) + .15 * t.unsqueeze(-1))


def mean_valid_loss(p,t,m):
    e=torch.where(m, p.float()-t.float(), 0)
    return e.square().sum() / m.sum().clamp_min(1)


@pytest.mark.parametrize('convention',[FlowConvention.NOISE_AT_ZERO,FlowConvention.NOISE_AT_ONE])
def test_reference_optional_lsd_has_teacher_student_roles(convention):
    torch.manual_seed(2)
    f=TwoTimeField()
    actions=torch.randn(3,8,2)
    noise=torch.randn_like(actions)
    # Native direction: progress 0 -> 1 is inverted for NOISE_AT_ONE.
    s=torch.full((3,8), .25 if convention is FlowConvention.NOISE_AT_ZERO else .75)
    t=torch.full((3,8), .65 if convention is FlowConvention.NOISE_AT_ZERO else .35)
    loss,info=lsd_loss(
        f, noise=noise,actions=actions,source_time=s,target_time=t,
        convention=convention,fm_reduce=mean_valid_loss,lsd_reduce=mean_valid_loss,
        fd_fraction=.01,deterministic=True,lsd_weight=.3,
    )
    assert len(f.calls)==5
    assert f.calls[:2]==[(False,actions.shape)]*2
    assert all(x[0] is True for x in f.calls[2:])
    assert info['path']=='straight' # Explicit alternative, NOT original stencil
    assert torch.isfinite(loss)
    loss.backward()
    assert f.weight.grad is not None and f.weight.grad.isfinite()


def test_reference_lsd_instant_fast_path():
    f=TwoTimeField()
    actions=torch.randn(2,8,2)
    noise=torch.randn_like(actions)
    t=torch.full((2,8),.4)
    loss,info=lsd_loss(f, noise=noise,actions=actions,source_time=t,target_time=t,
        convention=FlowConvention.NOISE_AT_ZERO,fm_reduce=mean_valid_loss,
        lsd_reduce=mean_valid_loss,lsd_weight=0)
    assert len(f.calls)==1
    assert info['lsd_loss']==0
    assert torch.isfinite(loss)


def test_zero_interval_conditioning_is_not_automatic():
    """Minimal model proves why the user's original skip_zero_interval_embedding=False matters."""
    hidden=torch.zeros(2,3,4)
    time_only=torch.ones_like(hidden)*.25
    residual_embedding=torch.ones_like(hidden)*.125
    ungated=time_only+residual_embedding
    # The original DiT's default can contribute a nonzero learned zero-time embedding.
    assert not torch.equal(ungated,time_only)
    gap=torch.zeros(2,3,1)
    corrected=time_only + gap*residual_embedding
    assert torch.equal(corrected,time_only)
