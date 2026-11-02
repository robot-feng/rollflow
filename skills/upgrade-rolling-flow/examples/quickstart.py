"""可执行尝鲜：真实的小型联合注意力头，非任何已发布 VLA 的替代测试。

不需要下载模型，展示严格加载、标量/逐位置时间等价、一次 grouped 训练
和 H64/C16 的 cold-4 / warm-1。数据是合成的，结果不能作为任务质量证据。
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assets"))
import torch
from torch import nn
from configuration import Selection, validate_resolved
from rolling_flow_torch import FlowConvention, RollingFlow, sample_grouped_time


class NativeScalarHead(nn.Module):
    """仅接受原生标量时间的基准；模块和参数在适配后保持相同。"""
    def __init__(self, horizon=64, action_dim=3, width=24):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.action_in = nn.Linear(action_dim, width)
        self.time_in = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.obs_in = nn.Linear(2, width)
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]), persistent=False)
        self.positions = nn.Parameter(torch.zeros(1, horizon, width))
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))
        self.out = nn.Linear(width, action_dim)

    def encode_time(self, time, batch, horizon):
        if time.shape != (batch,):
            raise ValueError("原生基准只接收 [B]")
        phase = time[..., None] * self.frequencies
        emb = self.time_in(torch.cat((phase.sin(), phase.cos()), -1))
        return emb[:, None].expand(batch, horizon, -1)

    def forward(self, x, time, observation):
        if x.ndim != 3 or x.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("动作形状不匹配")
        batch, horizon, _ = x.shape
        if observation.shape != (batch, 2):
            raise ValueError("观测应为 [B,2]")
        h = self.action_in(x) + self.encode_time(time, batch, horizon)
        h = h + self.positions + self.obs_in(observation)[:, None]
        n = self.norm(h)
        h = h + self.attn(n, n, n, need_weights=False)[0]
        return self.out(h + self.ff(h))


class TokenTimeHead(NativeScalarHead):
    """只覆盖形状管线；H 始终是 attention 序列轴，不扩成模型 batch。"""
    def encode_time(self, time, batch, horizon):
        if time.shape == (batch,):
            return super().encode_time(time, batch, horizon)
        if time.shape != (batch, horizon):
            raise ValueError("时间必须为 [B] 或 [B,H]")
        phase = time[..., None] * self.frequencies
        return self.time_in(torch.cat((phase.sin(), phase.cos()), -1))


def run_smoke(steps=24, horizon=64, execution_chunks=4, seed=7):
    config = Selection(horizon=horizon, execution_chunks=execution_chunks, initialization="scratch",
                       native_time_convention="noise_at_one", policy_name="tiny_demo").resolve()
    validate_resolved(config)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    baseline = NativeScalarHead(horizon=horizon)
    model = TokenTimeHead(horizon=horizon)
    model.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=True)
    x, obs, time = torch.randn(3,horizon,3), torch.randn(3,2), torch.rand(3)
    a = baseline(x,time,obs)
    b = model(x,time[:,None].expand(-1,horizon),obs)
    error = float((a-b).detach().abs().max())
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    a.square().mean().backward();b.square().mean().backward()
    gradient_error = max(float((p.grad-q.grad).abs().max()) for p,q in zip(baseline.parameters(),model.parameters(),strict=True))
    for p,q in zip(baseline.parameters(),model.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    group_rng=torch.Generator().manual_seed(seed+100)
    native_sampler=lambda n: torch.distributions.Beta(1.5,1.0).sample((n,))*.999+.001
    losses=[]
    for _ in range(steps):
        observation=torch.randn(4,2)
        pos=torch.arange(horizon)[None,:,None]/horizon
        actions=torch.sin(3*pos+observation[:,None,:1]).expand(-1,-1,3)+.05*torch.randn(4,horizon,3)
        noise=torch.randn_like(actions)
        t=sample_grouped_time(native_sampler,batch_size=4,horizon=horizon,
                              convention=FlowConvention.NOISE_AT_ONE,group_generator=group_rng)
        noisy=t[...,None]*noise+(1-t[...,None])*actions
        prediction=model(noisy,t,observation)
        loss=(prediction-(noise-actions)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError("训练出现非有限 loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("训练出现非有限梯度")
        optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    c=config['geometry']['actions_per_output']
    state=RollingFlow(action_dim=3,horizon=horizon,execution_horizon=c,convention=FlowConvention.NOISE_AT_ONE)
    nfe=[];shapes=[]
    for cycle in range(3):
        observation=torch.tensor([[cycle*.02,.1]])
        seen=[]
        def predict(latent,t):
            seen.append(list(latent.shape))
            return model(latent,t,observation)
        out=state.step(predict,noise=torch.randn(1,state.noise_horizon,3),
                       consumed_actions=0 if cycle==0 else c,
                       stream_ids=("demo",),context_key="episode1:model1:norm1:frame1")
        if out.shape!=(1,c,3) or not torch.isfinite(out).all():raise AssertionError("输出错误")
        assert len(seen)==(execution_chunks if cycle==0 else 1)
        assert all(shape==[1,horizon,3] for shape in seen)
        nfe.append(state.last_nfe);shapes.append(list(out.shape))
    state.reset()
    return {"scope":"synthetic_joint_attention_head_only", "seed":seed,"updates":steps,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "new_parameters":0,"scalar_output_max_error":error,
            "scalar_gradient_max_error":gradient_error,"loss_first":losses[0] if losses else None,
            "loss_last":losses[-1] if losses else None,"all_losses":losses,
            "cycle_nfe":nfe,"output_shapes":shapes,"reset_noise_horizon":state.noise_horizon,
            "quality_claim":"none; smoke/equivalence only, not pretrained VLA or closed-loop"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.steps<0:p.error('steps 必须非负')
    result=run_smoke(a.steps)
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text+'\n',encoding='utf8')
    print(text)


if __name__=='__main__':main()
