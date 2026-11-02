"""User selection, defaults, serialization and fail-closed geometry checks."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from configuration import Selection, validate_resolved, divisors
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from configure import parser, select_interactively


def test_quick_defaults_exact_user_choice():
    c=Selection().resolve();validate_resolved(c)
    assert c['geometry']==dict(horizon=64,execution_chunks=4,actions_per_output=16)
    assert c['model']['velocity_mode']=='instant'
    assert c['inference']['cold_nfe']==4 and c['inference']['warm_nfe']==1
    assert c['training']['group_choices']==[1,2,4,8,16,32,64]
    assert all(abs(p-1/7)<1e-12 for p in c['training']['group_probabilities'])
    assert not c['training']['cfg'] and not c['training']['lsd']['enabled']
    assert c['training']['lsd']['w_lsd']==0 and c['training']['ot']=='inherit_native'


@pytest.mark.parametrize('H,M,C',[(64,4,16),(32,4,8),(12,4,3),(50,5,10),(1,1,1),(6,3,2)])
def test_geometry_resolves_from_either_blocks_or_output(H,M,C):
    a=Selection(horizon=H,execution_chunks=M).resolve()
    b=Selection(horizon=H,actions_per_output=C).resolve()
    assert a==b
    validate_resolved(a)
    assert a['geometry']==dict(horizon=H,execution_chunks=M,actions_per_output=C)


@pytest.mark.parametrize('kw',[
    dict(horizon=0),dict(horizon=True),dict(horizon=64.),dict(execution_chunks=3),
    dict(actions_per_output=5),dict(execution_chunks=4,actions_per_output=8),
    dict(execution_chunks=False),dict(refinements=2),dict(cold_steps=1),
    dict(single_group_probability=float('nan')),dict(single_group_probability=1.1),
    dict(single_group_probability=True),dict(preset='meanflow'),dict(policy_name='../native'),
    dict(initialization='finetune'),dict(initialization='scratch',checkpoint_horizon=64),
    dict(native_time_convention='guessed'),dict(allow_horizon_change='yes'),
])
def test_bad_or_ambiguous_selection_rejected(kw):
    with pytest.raises(ValueError):Selection(**kw).resolve()


def test_checkpoint_default_preserves_length_not_64():
    c=Selection(initialization='finetune',checkpoint_horizon=12).resolve()
    assert c['geometry']==dict(horizon=12,execution_chunks=4,actions_per_output=3)
    validate_resolved(c)
    with pytest.raises(ValueError,match='旧权重'):
        Selection(initialization='finetune',checkpoint_horizon=12,horizon=64).resolve()


def test_50_requires_deliberate_geometry_selection():
    with pytest.raises(ValueError,match='可选块数'):
        Selection(initialization='finetune',checkpoint_horizon=50).resolve()
    c=Selection(initialization='finetune',checkpoint_horizon=50,execution_chunks=5).resolve()
    assert c['geometry']['actions_per_output']==10


def test_explicit_horizon_change_is_warning_not_strict_compatibility_claim():
    c=Selection(checkpoint_horizon=12,horizon=64,allow_horizon_change=True).resolve()
    validate_resolved(c)
    assert any('专项' in w or '单独验证' in w for w in c['warnings'])
    assert c['integration_status']=='configuration_only_not_adapted'


@pytest.mark.parametrize('recipe,pfm,warmup',[('core',.7,20000),('starvla-wrapper',.3,5000)])
def test_lsd_requires_explicit_choice_and_preserves_recipe(recipe,pfm,warmup):
    c=Selection(preset='lsd-original',original_recipe=recipe).resolve();validate_resolved(c)
    assert c['training']['group_choices']==[1,2,4]
    assert c['training']['lsd']['p_fm']==pfm
    assert c['training']['lsd']['fm_curriculum_steps']==warmup
    assert c['inference']['cold_nfe']==1
    assert c['training']['lsd']['variant']=='original_absolute_fd_v1'
    with pytest.raises(ValueError):Selection(preset='lsd-original',group_domain='actions').resolve()


def test_lsd_budget_not_arbitrary_original_executor():
    c=Selection(preset='lsd-original',refinements=2,cold_steps=2).resolve();validate_resolved(c)
    assert c['inference']['iterative_cold_start']
    with pytest.raises(ValueError):Selection(preset='lsd-original',refinements=4,cold_steps=3).resolve()


@pytest.mark.parametrize('mutation',[
    lambda c:c['geometry'].update(actions_per_output=8),
    lambda c:c['training']['lsd'].update(enabled=True),
    lambda c:c['training'].update(cfg=True),
    lambda c:c['inference'].update(warm_nfe=4),
    lambda c:c['model'].update(new_trainable_parameters='arbitrary'),
    lambda c:c.update(registered_policy=c['baseline_policy']),
    lambda c:c.update(integration_status='production_passed'),
    lambda c:c['training'].update(one_layout_per_batch=1),
])
def test_edited_resolved_config_cannot_silently_change_contract(mutation):
    c=Selection().resolve();mutation(c)
    with pytest.raises(ValueError):validate_resolved(c)


def test_roundtrip_json():
    a=Selection().resolve()
    b=json.loads(json.dumps(a,allow_nan=False))
    validate_resolved(b);assert a==b


def test_interactive_enter_is_quick_default():
    a=parser().parse_args(['--interactive'])
    answers=iter(['','',''])
    select_interactively(a,lambda _:next(answers))
    assert a.preset=='quick' and a.horizon==64 and a.execution_chunks==4


def test_interactive_lsd_is_opt_in_and_overrides_budget():
    a=parser().parse_args(['--interactive'])
    answers=iter(['2','64','4','core','2','2'])
    select_interactively(a,lambda _:next(answers))
    assert a.preset=='lsd-original' and a.refinements==2 and a.cold_steps==2


def test_cli_writes_config_without_mutating_or_overwriting(tmp_path):
    output=tmp_path/'my_config.json'
    args=[sys.executable,str(ROOT/'scripts/configure.py'),'--yes','--output',str(output)]
    a=subprocess.run(args,capture_output=True,text=True)
    assert a.returncode==0,a.stderr
    before=output.read_bytes();validate_resolved(json.loads(before))
    b=subprocess.run(args,capture_output=True,text=True)
    assert b.returncode!=0 and output.read_bytes()==before
    assert list(tmp_path.iterdir())==[output]
