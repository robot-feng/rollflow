"""Matched LSD delta or weight experiments; sequential, restart-safe outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sweep',choices=['delta','weight'],default='delta')
    parser.add_argument('--dataset',default='rollflow/pendulum-mpc-v0')
    parser.add_argument('--max-episodes',type=int,default=100)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--eval-max-steps',type=int,default=200)
    parser.add_argument('--horizon',type=int,default=32)
    parser.add_argument('--chunk',type=int,default=8)
    parser.add_argument('--nfe',type=int,default=4)
    parser.add_argument('--eval-nfe',type=int,nargs='+',default=[1,2,4])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'suite.json'
    if manifest_path.exists():raise FileExistsError(manifest_path)
    # Match initial gradient magnitudes approximately, rather than keeping a
    # coefficient that could make small delta dominate and trigger clipping.
    arms=([('delta_010',.01,.05),('delta_005',.005,.027),('delta_001',.001,.0065)]
          if args.sweep=='delta' else
          [('weight_005',.01,.05),('weight_020',.01,.2),('weight_050',.01,.5)])
    status=dict(pid=os.getpid(),started=time.time(),runs=[],
        note=('LSD-only diagnostic; matching weights are estimates from a few batches, not proof of equal gradient strength'
              if args.sweep=='delta' else 'LSD-only diagnostic; same delta and differing loss weights'))
    manifest_path.write_text(json.dumps(status,indent=2))
    for name,delta,weight in arms:
        command=[sys.executable,str(Path(__file__).with_name('run.py')),'train',
            '--output',str(args.output/name),'--steps',str(args.steps),'--seed',str(args.seed),
            '--dataset',args.dataset,'--max-episodes',str(args.max_episodes),
            '--horizon',str(args.horizon),'--chunk',str(args.chunk),'--nfe',str(args.nfe),
            '--variant','rolling_lsd','--precision','fp32','--fm-warmup','2000','--lsd-ramp','1000',
            '--batch','8','--repeat','4','--width',str(args.width),'--layers','2','--lr','.0003',
            '--delta',str(delta),'--w-lsd',str(weight),'--log-every','250','--save-every','2000',
            '--eval-nfe',*[str(value) for value in args.eval_nfe],
            '--eval-episodes','10','--eval-max-steps',str(args.eval_max_steps)]
        row=dict(name=name,delta=delta,w_lsd=weight,command=command,status='running',started=time.time())
        status['runs'].append(row)
        with (args.output/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=process.pid;manifest_path.write_text(json.dumps(status,indent=2))
            exit_code=process.wait()
        row.update(status='completed' if exit_code==0 else 'failed',exit_code=exit_code,finished=time.time())
        manifest_path.write_text(json.dumps(status,indent=2))
        print(name,row['status'],flush=True)
    status['finished']=time.time();manifest_path.write_text(json.dumps(status,indent=2))
    if any(row['exit_code'] for row in status['runs']):raise SystemExit(1)


if __name__=='__main__':main()
