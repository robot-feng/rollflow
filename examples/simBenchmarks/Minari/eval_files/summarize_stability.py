"""Summarize sampled training logs; logged points are not all optimizer steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths, *, after=30000, interval=5000, clip=1.):
    if interval <= 0 or clip <= 0:
        raise ValueError('interval and clip must be positive')
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            data = json.loads(line)
            step = int(data['step'])
            if step in rows:
                raise ValueError(f'duplicate logged step {step}')
            for key in ('grad_norm', 'loss', 'validation_fm_mse'):
                if not np.isfinite(float(data[key])):
                    raise ValueError(f'nonfinite {key} at step {step}')
            rows[step] = data
    groups = {}
    for step, row in sorted(rows.items()):
        if step <= after:
            continue
        bucket = (step-after-1)//interval
        groups.setdefault(bucket, []).append(row)
    result=[]
    for bucket, values in sorted(groups.items()):
        grad=np.array([v['grad_norm'] for v in values],dtype=float)
        validation=np.array([v['validation_fm_mse'] for v in values],dtype=float)
        result.append(dict(start=after+bucket*interval+1,end=after+(bucket+1)*interval,
            last_logged_step=values[-1]['step'],logged_points=len(values),
            grad_median=float(np.median(grad)),grad_p95=float(np.quantile(grad,.95)),
            grad_max=float(grad.max()),logged_grad_above_clip=int((grad>clip).sum()),
            validation_fm_min=float(validation.min()),validation_fm_max=float(validation.max())))
    if not result:
        raise ValueError('no logged points after selected step')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs='+',required=True,help='label=metrics.jsonl[,continuation.jsonl]')
    p.add_argument('--after',type=int,default=30000)
    p.add_argument('--interval',type=int,default=5000)
    p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result={}
    for item in a.run:
        label,sep,files=item.partition('=')
        if not sep or not label or label in result:raise ValueError('run labels must be unique and nonempty')
        paths=[Path(x) for x in files.split(',')]
        result[label]={'metrics_files':[str(x) for x in paths],
            'windows':summarize(paths,after=a.after,interval=a.interval,clip=a.clip)}
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Training stability in sampled logs','',
           'Only recorded log points are summarized; counts are not the number of all training steps.',
           'Gradient norms are before clipping; validation FM MSE does not measure mean-velocity rollout.',
           'Loss scales can differ by intervention. An incomplete final window is not evidence of convergence.','',
           '|Run|Step window|Last logged|N|Grad median|Grad p95|Grad max|Grad > clip|Validation FM MSE min/max|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,summary in result.items():
        for w in summary['windows']:
            lines.append(f"|{label}|{w['start']}–{w['end']}|{w['last_logged_step']}|{w['logged_points']}|"
                f"{w['grad_median']:.3f}|{w['grad_p95']:.3f}|{w['grad_max']:.3f}|"
                f"{w['logged_grad_above_clip']}|{w['validation_fm_min']:.5f} / {w['validation_fm_max']:.5f}|")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(a.output/'report.md')


if __name__=='__main__':main()
