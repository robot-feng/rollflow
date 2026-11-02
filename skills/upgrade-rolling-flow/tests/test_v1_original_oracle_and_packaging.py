"""Independent loss/gradient oracle and release-document checks."""
import copy
import importlib.util
from pathlib import Path
import re
import sys

import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('rollflow_original_v1_oracle',ROOT/'references/original_sources/rolling_meanflow.py')
original=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=original;SPEC.loader.exec_module(original)


class JointField(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor(.37));self.calls=[]
    def forward(self,x,s,t,context=None):
        self.calls.append((len(x),torch.is_grad_enabled()))
        return self.w*torch.tanh(x+.15*x.mean(1,keepdim=True)+.1*s+.2*t+context[:,:1,None])


def test_packed_original_matches_sequential_loss_parameter_and_context_gradients():
    torch.manual_seed(70)
    x0=torch.randn(3,8,2);x1=torch.randn_like(x0)
    s=torch.full((3,8,1),.2);t=torch.full_like(s,.7);s[1]=t[1]
    context=torch.randn(3,2,requires_grad=True)
    model=JointField();other=copy.deepcopy(model);ctx2=context.detach().clone().requires_grad_()
    loss,_=original.central_difference_lsd(model,x0,x1,s,t,delta=.01,context=context,w_lsd=.5)
    loss.backward()
    rows=torch.tensor([True,False,True])
    # Independent sequential implementation of the USER'S absolute-FD definition.
    xs=(1-s)*x0+s*x1;xt=(1-t)*x0+t*x1
    ss=s[rows];tt=t[rows];xx=xs[rows]
    with torch.no_grad():
        mapped=xx+(tt-ss)*other(xx,ss,tt,ctx2[rows])
        teacher=other(mapped,tt,tt,ctx2[rows])
    minus=xx+(tt-.01-ss)*other(xx,ss,tt-.01,ctx2[rows])
    plus=xx+(tt+.01-ss)*other(xx,ss,tt+.01,ctx2[rows])
    local=other(xt,t,t,ctx2)
    expected=(local-(x1-x0)).square().mean()+.5*((plus-minus)/.02-teacher).square().mean()
    expected.backward()
    assert model.calls==[(2,False),(2,False),(7,True)]
    torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(model.w.grad,other.w.grad,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(context.grad,ctx2.grad,atol=1e-6,rtol=1e-5)


def test_skill_metadata_and_all_relative_links():
    text=(ROOT/'SKILL.md').read_text(encoding='utf8')
    assert text.startswith('---\nname: upgrade-rolling-flow\n')
    assert '  version: "1.0.0"' in text
    assert ROOT.name=='upgrade-rolling-flow'
    front=text.split('---',2)[1]
    allowed={'name','description','compatibility','metadata','license','allowed-tools'}
    keys=set(re.findall(r'^([a-z-]+):',front,re.M))
    assert keys<=allowed
    assert len(text.splitlines())<500
    for path in [ROOT/'SKILL.md',ROOT/'README.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
            if '://' not in link:
                assert (ROOT/link.split('#')[0]).exists(),link


def test_scanner_does_not_follow_external_file_symlink(tmp_path):
    from scan_policy import scan
    outside=tmp_path/'secret.py';outside.write_text('v_gt = actions - noise\n')
    repo=tmp_path/'repo';repo.mkdir()
    try:(repo/'linked.py').symlink_to(outside)
    except OSError:pytest.skip('symlinks unavailable on this host')
    assert scan(repo)['count']==0


def test_host_report_starts_unverified():
    import json
    report=json.loads((ROOT/'templates/HOST_VALIDATION.json').read_text())
    assert report['evidence_level']=='NOT_RUN'
    assert all(s=='NOT_RUN' for s in report['checks'].values())


def test_original_integrity_check_still_rejects_tampering_under_python_O(tmp_path):
    import shutil,subprocess
    shutil.copytree(ROOT/'references',tmp_path/'references')
    (tmp_path/'tools').mkdir()
    shutil.copy2(ROOT/'tools/check_originals.py',tmp_path/'tools/check_originals.py')
    source=tmp_path/'references/original_sources/rolling_meanflow.py'
    source.write_bytes(source.read_bytes()+b'\n# injected change\n')
    p=subprocess.run([sys.executable,'-O',str(tmp_path/'tools/check_originals.py')],capture_output=True,text=True)
    assert p.returncode!=0 and 'changed' in p.stderr
