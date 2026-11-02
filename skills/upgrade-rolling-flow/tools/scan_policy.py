"""Non-mutating first-pass scout for an unfamiliar robot action-policy repository.

This is a *candidate locator*, never a framework compatibility verdict.
Do not turn its regex evidence into automatic source rewrites.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    'flow_target': r'flow_matching|target_velocity|noise\s*-\s*actions|actions\s*-\s*noise|v_gt\s*=|u_t\s*=',
    'time_sampler': r'sample_time\s*\(|sample_timestep|timestep_sampler|Beta\(|time_sampling',
    'time_encoder': r'time_mlp|timestep_encoder|time_embedding|time_emb\b|adaln|adarms|AdaLayerNorm',
    'sampler': r'euler_integrate|denoise_step|sample_actions|inference_action|flow_map\s*\(',
    'registry': r'register_policy|get_action_model|register\s*\(|model_type\s*=|policy_name',
    'rtc_prefix': r'rtc_processor|RTCProcessor|prefill_actions|inference_delay|action_prefix_mask',
    'mask': r'action_is_pad|action_dim_mask|padding_mask|valid_mask|action_mask',
    'state': r'past_key_values|cache_position|reset_cache|_action_queue|action_cache',
}
EXCLUDE={'__pycache__','.git','.venv','venv','site-packages','node_modules','.mypy_cache'}


def scan(root:Path,*,max_bytes:int=3_000_000):
    root = root.resolve()
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    if not root.is_dir():
        raise NotADirectoryError(root)
    found=[]
    for p in root.rglob('*.py'):
        if p.is_symlink() or root not in p.resolve().parents:
            continue
        if any(part in EXCLUDE for part in p.relative_to(root).parts):
            continue
        try:
            if p.stat().st_size>max_bytes:
                continue
            source=p.read_text(encoding='utf8')
        except (OSError,UnicodeError):
            continue
        hits={name:len(re.findall(pattern,source,flags=re.IGNORECASE)) for name,pattern in PATTERNS.items()}
        if not any(hits.values()):
            continue
        found.append({'file':str(p.relative_to(root)),'hits':{k:v for k,v in hits.items() if v},
                      'score':sum(hits.values()), 'bytes':p.stat().st_size})
    found.sort(key=lambda r:(-r['score'],r['file']))
    return {
       'root':str(root.resolve()),
       'count':len(found),
       'files':found,
       'WARNING':'Regex locations only; confirm actual selected factory/config branch and the exact training/inference formulas manually.',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    report=scan(a.repo)
    value=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(value+'\n',encoding='utf8')
    print(value if not a.output else f"Found {report['count']} candidate files; wrote {a.output}")


if __name__=='__main__':
    main()
