import importlib.util
import sys
from pathlib import Path
import torch
import pytest
from scan_policy import scan

SOURCE=Path(__file__).resolve().parents[1]/'references/original_sources/rolling_meanflow.py'
spec=importlib.util.spec_from_file_location('original_rollflow_full',SOURCE)
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)

class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.6))
        self.call_logs=[]
    def forward(self,x,s,t,context=None,**kwargs):
        self.call_logs.append((torch.is_grad_enabled(), x.shape, s.shape,t.shape))
        return self.weight*torch.ones_like(x)


def test_original_full_training_fast_and_lsd_mode():
    a=torch.randn(4,64,2)
    cfg=original.RollFlowConfig(horizon=64,chunk_size=16,action_dim=2,use_ot=False,
        w_fm=1.,w_lsd=.25,p_fm=0.0,fm_curriculum_steps=0,total_train_steps=1)
    engine=original.RollFlow(cfg)
    model=Recorder()
    loss,stats=engine.loss(model,a,step=2)
    assert torch.isfinite(loss) and stats['num_time_groups'] in (1,2,4)
    assert len(model.call_logs) in (1,3) # short intervals/near endpoints can disable LSD
    loss.backward()
    assert model.weight.grad is not None and model.weight.grad.isfinite()
    cfg_fm=original.RollFlowConfig(horizon=64,chunk_size=16,action_dim=2,use_ot=False,
        w_fm=1.,w_lsd=0.,p_fm=1.)
    fm=original.RollFlow(cfg_fm)
    m2=Recorder()
    fm_loss,meta=fm.loss(m2,a,step=0)
    assert len(m2.call_logs)==1
    assert torch.isfinite(fm_loss) and meta['lsd_loss']==0


def test_original_cache_warm_and_reset_calls():
    cfg=original.RollFlowConfig(horizon=64,chunk_size=16,action_dim=2,use_ot=False,
        iterative_cold_start=True,velocity_mode='instant')
    engine=original.RollFlow(cfg)
    m=Recorder()
    out=engine.step(m,batch=1,device='cpu',dtype=torch.float32)
    assert out.shape==(1,16,2)
    assert len(m.call_logs)==4
    engine.step(m,batch=1,device='cpu',dtype=torch.float32)
    assert len(m.call_logs)==5
    assert engine.cache_info.shape==(1,64,2)
    engine.reset()
    assert engine.cache_info is None


def test_original_empty_padding_mask_finite():
    m=Recorder(); z=torch.zeros(3,8,2)
    t=torch.full((3,8,1),.5)
    pad=torch.ones((3,8),dtype=torch.bool)
    loss,st=original.central_difference_lsd(m,z,z,t,t,delta=.01,pad=pad)
    assert loss.item()==0 and torch.isfinite(loss)


def test_original_lsd_teacher_param_has_no_gradient():
    class Split(torch.nn.Module):
        def __init__(self):
            super().__init__();self.teacher=torch.nn.Parameter(torch.tensor(2.));self.student=torch.nn.Parameter(torch.tensor(.2))
        def forward(self,x,s,t,context=None,**kwargs):
            if torch.is_grad_enabled():
                return self.student*(x+1)
            return self.teacher*(x+1)
    m=Split()
    x0=torch.zeros(2,4,1);x1=torch.ones_like(x0)
    s=torch.full((2,4,1),.2);t=torch.full((2,4,1),.8)
    loss,_=original.central_difference_lsd(m,x0,x1,s,t,delta=.01,w_lsd=1)
    loss.backward()
    assert m.teacher.grad is None
    assert m.student.grad is not None and torch.isfinite(m.student.grad)


def test_original_tiny_interval_skips_lsd_without_stealing_fm():
    m=Recorder();x0=torch.zeros(2,4,1);x1=torch.ones_like(x0)
    s=torch.full((2,4,1),.495);t=torch.full((2,4,1),.50)
    loss,meta=original.central_difference_lsd(m,x0,x1,s,t,delta=.01,w_lsd=.4)
    assert len(m.call_logs)==1
    assert meta['lsd_loss']==0
    assert meta['fm_loss']>0


def test_scout_only_reports_candidates(tmp_path):
    (tmp_path/'policy.py').write_text('def forward():\n    time = sample_time(1, device)\n    return actions - noise\n')
    (tmp_path/'other.py').write_text('def other():\n    return 1\n')
    report=scan(tmp_path)
    assert report['count']==1
    assert report['files'][0]['file']=='policy.py'
    assert {'flow_target','time_sampler'} <= set(report['files'][0]['hits'])
    assert 'confirm actual selected' in report['WARNING']


def test_independent_adapter_does_not_mutate_baseline():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'examples'))
    from independent_instant_adapter import RollFlowInstantAdapter
    from rolling_flow_torch import FlowConvention
    baseline_calls=[]
    def baseline(x,t,obs):
        baseline_calls.append((x.shape,t.shape,obs))
        return torch.ones_like(x)
    adapter=RollFlowInstantAdapter(baseline,action_dim=2,horizon=8,execution_horizon=2,
        convention=FlowConvention.NOISE_AT_ZERO)
    out=adapter.next_chunk('obs-1',torch.zeros(1,8,2),consumed_actions=0,stream_ids=('arm',),session_key='s')
    assert out.shape==(1,2,2)
    assert len(baseline_calls)==4
    assert all(o=='obs-1' for _,_,o in baseline_calls)
    adapter.next_chunk('obs-2',torch.zeros(1,2,2),consumed_actions=2,stream_ids=('arm',),session_key='s')
    assert len(baseline_calls)==5 and baseline_calls[-1][-1]=='obs-2'
    adapter.reset()
    assert adapter.state.noise_horizon==8
