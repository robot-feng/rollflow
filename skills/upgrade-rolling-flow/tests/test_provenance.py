from pathlib import Path
import ast
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def test_originals_unchanged():
    manifest = json.loads((ROOT/'references/original_manifest.json').read_text())
    for name, record in manifest['sources'].items():
        p=ROOT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        assert len(p.read_bytes())==record['bytes']
        compile(p.read_bytes(),str(p),'exec')


def test_original_docstring_embedded_verbatim():
    file=ROOT/'references/original_sources/rolling_meanflow.py'
    doc=ast.get_docstring(ast.parse(file.read_text()),clean=False)
    assert f'```text\n{doc}\n```' in (ROOT/'references/ORIGINAL_ROLLFLOW_ALGORITHM.md').read_text()
    assert 'There is no JVP, CSF, Split loss' in doc
    assert 'Div(M)' in doc and 'M = H / C' in doc


def test_original_dit_has_default_ungated_interval_and_opt_in_gate():
    """This is a SOURCE audit; not a runtime proof of diffusers' actual embedding."""
    p=ROOT/'references/original_sources/cross_attention_dit.py'
    t=ast.parse(p.read_text())
    dit=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='DiT')
    init=next(n for n in dit.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    j=init.args.args.index(next(a for a in init.args.args if a.arg=='skip_zero_interval_embedding'))
    default=init.args.defaults[j-(len(init.args.args)-len(init.args.defaults))]
    assert isinstance(default,ast.Constant) and default.value is False
    forward=next(n for n in dit.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert 'skip_zero_interval_embedding' in ast.unparse(forward)


def test_original_action_head_has_separate_config_and_original_default():
    p=ROOT/'references/original_sources/starvla_rollflow_action_head.py'
    t=ast.parse(p.read_text())
    c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='RollFlowActionHeadConfig')
    init=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    # Do not silently unify p_fm config defaults across adapters/core.
    names=[a.arg for a in init.args.args]
    i=names.index('p_fm')-len(names)+len(init.args.defaults)
    assert init.args.defaults[i].value==0.3
    r=ROOT/'references/original_sources/rolling_meanflow.py'
    n=ast.parse(r.read_text())
    cls=next(n for n in n.body if isinstance(n,ast.ClassDef) and n.name=='RollFlowConfig')
    assert any(isinstance(x,ast.AnnAssign) and getattr(x.target,'id',None)=='p_fm' and x.value.value==0.7 for x in cls.body)
