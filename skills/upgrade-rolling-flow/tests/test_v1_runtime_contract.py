"""Fresh audits of the reference kernel and a genuinely joint small head."""
import copy
import io
from pathlib import Path
import sys
import pytest
import torch
from rolling_flow_torch import FlowConvention, RollingFlow, sample_grouped_time
from lsd_flow_torch import RollingLSD, EndpointResidual, require_deterministic_modules
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from quickstart import NativeScalarHead,TokenTimeHead,run_smoke


def test_model_shape_change_strict_checkpoint_and_no_parameter_change():
    torch.manual_seed(3)
    a=NativeScalarHead(horizon=12);b=TokenTimeHead(horizon=12)
    b.load_state_dict(a.state_dict(),strict=True)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    x=torch.randn(2,12,3);t=torch.tensor([.2,.7]);obs=torch.randn(2,2)
    ya=a(x,t,obs);yb=b(x,t[:,None].expand(-1,12),obs)
    torch.testing.assert_close(ya,yb,atol=1e-6,rtol=1e-5)
    ya.square().mean().backward();yb.square().mean().backward()
    for p,q in zip(a.parameters(),b.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    stream=io.BytesIO();torch.save(b.state_dict(),stream);stream.seek(0)
    c=TokenTimeHead(horizon=12);c.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    torch.testing.assert_close(b(x,t,obs),c(x,t,obs),atol=0,rtol=0)


@pytest.mark.parametrize('g',[1,2,4,8,16,32,64])
def test_joint_head_all_h64_groupings_forward_backward(g):
    torch.manual_seed(g)
    model=TokenTimeHead()
    x=torch.randn(2,64,3);obs=torch.randn(2,2)
    t=sample_grouped_time(lambda n:torch.rand(n),batch_size=2,horizon=64,
                          num_groups=g,convention=FlowConvention.NOISE_AT_ZERO)
    batches=[]
    hook=model.attn.register_forward_pre_hook(lambda m,args:batches.append(args[0].shape))
    y=model(x,t,obs);y.square().mean().backward();hook.remove()
    assert batches==[torch.Size([2,64,24])]
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    if g>1:
        assert not torch.allclose(y,model(x,t[:,:1].expand_as(t),obs))


def test_configurable_group_domain_and_weight():
    for domain_unit in (1,16):
        t=sample_grouped_time(lambda n:torch.linspace(.1,.8,n),batch_size=3,horizon=64,
                              group_unit=domain_unit,single_group_probability=1,
                              convention=FlowConvention.NOISE_AT_ONE)
        assert torch.equal(t,t[:,:1].expand_as(t))
    with pytest.raises(ValueError):
        sample_grouped_time(torch.rand,batch_size=2,horizon=64,group_unit=16,
                            num_groups=64,convention=FlowConvention.NOISE_AT_ZERO)


@pytest.mark.parametrize('conv',list(FlowConvention))
def test_cold_staircase_trace_and_warm_shift(conv):
    r=RollingFlow(action_dim=1,horizon=8,execution_horizon=2,convention=conv)
    times=[]
    def field(x,t):times.append(t.clone());return torch.zeros_like(x)
    noise=torch.arange(8,dtype=torch.float32)[None,:,None]
    y=r.step(field,noise=noise,context_key='a')
    expected=torch.tensor([[0,0,0,0],[.25,0,0,0],[.5,.25,0,0],[.75,.5,.25,0]])
    if conv is FlowConvention.NOISE_AT_ONE:expected=1-expected
    assert torch.equal(torch.cat(times)[:,::2],expected)
    y.fill_(100) # returned actions must not alias retained cache
    assert torch.equal(r._cache,noise)
    z=r.step(field,noise=torch.full((1,2,1),9.),consumed_actions=2,context_key='a')
    assert torch.equal(z,noise[:,2:4])
    assert torch.equal(r._cache[:,-2:],torch.full((1,2,1),9.))
    assert r.last_nfe==1


@pytest.mark.parametrize('cls',[RollingFlow,RollingLSD])
def test_numeric_masks_fail_without_changing_state(cls):
    r=cls(action_dim=2,horizon=8,execution_horizon=2,convention=FlowConvention.NOISE_AT_ZERO)
    predict=lambda x,*t:torch.zeros_like(x)
    r.step(predict,noise=torch.zeros(1,8,2))
    before=r._cache.clone()
    with pytest.raises(TypeError):
        r.step(predict,noise=torch.zeros(1,2,2),consumed_actions=2,action_dim_mask=torch.tensor([[1.,float('nan')]]))
    assert torch.equal(before,r._cache)


@pytest.mark.parametrize('cls',[RollingFlow,RollingLSD])
def test_exception_does_not_commit_cache(cls):
    r=cls(action_dim=1,horizon=8,execution_horizon=2,convention=FlowConvention.NOISE_AT_ZERO)
    r.step(lambda x,*t:torch.zeros_like(x),noise=torch.zeros(1,8,1),context_key='episode1')
    before=r._cache.clone()
    def broken(x,*t):
        x.fill_(100)
        raise RuntimeError('injected field failure')
    with pytest.raises(RuntimeError):r.step(broken,noise=torch.zeros(1,2,1),consumed_actions=2,context_key='episode1')
    assert torch.equal(before,r._cache)
    with pytest.raises(ValueError):r.step(lambda x,*t:x,noise=torch.zeros(1,2,1),consumed_actions=2,context_key='episode2')
    assert torch.equal(before,r._cache)


def test_geometry_mutation_detected_even_after_reset():
    r=RollingFlow(action_dim=1,convention=FlowConvention.NOISE_AT_ZERO)
    r.steps=0;r.reset()
    with pytest.raises(ValueError):r.step(lambda x,t:x,noise=torch.zeros(1,64,1))


def test_zero_interval_residual_identity_after_parameter_training():
    m=EndpointResidual(8)
    c=torch.randn(2,5,8,requires_grad=True);s=torch.rand(2,5)*.2;t=s+.5
    before=m(c,s,t)
    torch.testing.assert_close(before,c,atol=0,rtol=0)
    opt=torch.optim.SGD(m.parameters(),lr=.1)
    loss=(m(c,s,t)-torch.ones_like(c)).square().mean();loss.backward();opt.step()
    assert m.output.weight.grad.abs().max()>0
    assert not torch.equal(m(c,s,t),c)
    torch.testing.assert_close(m(c,t,t),c,atol=0,rtol=0)


def test_no_grad_is_not_dropout_control():
    m=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.5)).train()
    with torch.no_grad(),pytest.raises(ValueError):require_deterministic_modules(m)
    m.eval();require_deterministic_modules(m)


def test_fast_start_end_to_end_synthetic_smoke():
    result=run_smoke(steps=4)
    assert result['cycle_nfe']==[4,1,1]
    assert result['output_shapes']==[[1,16,3]]*3
    assert result['new_parameters']==0
