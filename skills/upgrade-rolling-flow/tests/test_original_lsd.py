"""Execute the supplied original, not a reconstructed loss implementation."""
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'references/original_sources/rolling_meanflow.py'
spec=importlib.util.spec_from_file_location('user_original_rollflow',SOURCE)
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(0.35))
        self.calls=[]

    def forward(self,x,s,t,context=None,**kwargs):
        self.calls.append((x.shape[0],torch.is_grad_enabled()))
        return self.scale*(0.1*x+0.2*s+0.3*t).expand_as(x)


def test_original_group_candidates_are_div_H_over_C_not_div_H():
    cfg=original.RollFlowConfig(horizon=64,action_dim=2,chunk_size=16,use_ot=False)
    sampler=original.StaircaseTimeSampler(cfg)
    assert sampler._train_ks.tolist()==[1,2,4]


@pytest.mark.parametrize('g', [1,2,4])
def test_original_sampler_block_sizes(g):
    cfg=original.RollFlowConfig(horizon=64,action_dim=3,chunk_size=16,use_ot=False)
    data=original.StaircaseTimeSampler(cfg).sample_training(5,device='cpu',num_time_groups=g)
    assert data.s.shape==data.t.shape==(5,64,1)
    assert data.active.shape==(5,64,1)
    assert data.num_time_groups==g
    assert data.block_size==4//g
    grouped=data.t.squeeze(-1).reshape(5,g,64//g)[:,:,0]
    assert torch.all(grouped[:,:-1]>=grouped[:,1:])
    assert torch.all(data.s<=data.t) and torch.all(data.s>=0) and torch.all(data.t<=1)
    assert torch.allclose(data.s,data.t*data.ratio)


def test_original_mandatory_teacher_and_packed_student_gradients():
    torch.manual_seed(42)
    m=Recorder()
    x0=torch.randn(3,8,2)
    x1=torch.randn_like(x0)
    s=torch.full((3,8,1),0.23)
    t=torch.full((3,8,1),0.71)
    loss,stats=original.central_difference_lsd(m,x0,x1,s,t,delta=0.01,w_fm=1,w_lsd=.3)
    assert len(m.calls)==3
    assert m.calls[:2]==[(3,False),(3,False)]
    assert m.calls[2]==(9,True) # [LSD-, LSD+, full FM] = B*3
    assert torch.isfinite(loss)
    assert stats['lsd_loss']>0 and stats['fm_loss']>0
    loss.backward()
    assert m.scale.grad is not None and m.scale.grad.isfinite()


def test_original_lsd_off_one_forward_and_old_fm_objective():
    torch.manual_seed(11)
    m=Recorder()
    x0=torch.randn(2,8,3); x1=torch.randn_like(x0)
    t=torch.rand(2,8,1)
    loss,stats=original.central_difference_lsd(m,x0,x1,t,t,delta=.01,w_lsd=0)
    assert len(m.calls)==1
    expected=m((1-t)*x0+t*x1,t,t)
    reference=((expected-(x1-x0))**2).mean()
    assert torch.allclose(loss,reference,atol=1e-6)
    assert stats['lsd_loss']==0


def test_original_lsd_fm_only_due_to_zero_gap():
    m=Recorder()
    x0=torch.randn(2,6,2);x1=torch.randn_like(x0)
    t=torch.full((2,6,1),.5)
    loss,stats=original.central_difference_lsd(m,x0,x1,t,t,delta=.01,w_lsd=.3)
    assert len(m.calls)==1
    assert stats['lsd_loss']==0
    assert torch.isfinite(loss)


def test_original_absolute_stencil_differs_from_radial_when_coupled():
    """Keep original absolute t±δ and optional radial t±h*(t-s) distinct."""
    s=torch.tensor([0.0,0.0],dtype=torch.double)
    t=torch.tensor([0.6,0.4],dtype=torch.double)
    d=t-s
    def u(e): return torch.stack((1+.3*e[1],2+.1*e[0]))
    eps=1e-5
    def phi(e):return s+(e-s)*u(e)
    absolute=(phi(t+eps)-phi(t-eps))/(2*eps)
    radial=(phi(t+eps*d)-phi(t-eps*d))/(2*eps)/d
    # Due to other horizon time positions participating in attention, these differ.
    assert (absolute-radial).abs().max()>0.01
    # Both finite differences agree with their distinct analytic derivatives.
    assert torch.allclose(absolute,u(t)+d*torch.tensor([.3,.1],dtype=torch.double),atol=1e-8)
    assert torch.allclose(radial,u(t)+torch.tensor([.3*d[1],.1*d[0]]),atol=1e-8)


def test_source_core_keeps_native_no_jvp():
    text=SOURCE.read_text()
    assert 'def central_difference_lsd' in text
    assert 'torch.no_grad()' in text
    assert 'torch.func.jvp(' not in text
