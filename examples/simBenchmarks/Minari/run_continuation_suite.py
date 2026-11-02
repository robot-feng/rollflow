"""Continue each completed arm with matching optimizer/scheduler state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional-steps',type=int,default=10000)
    p.add_argument('--allow-resume-code-change',action='store_true')
    args=p.parse_args()
    if args.additional_steps<=0:p.error('additional-steps must be positive')
    original=json.loads((args.source/'suite.json').read_text())
    if not original.get('finished') or any(r['status']!='completed' for r in original['runs']):
        raise RuntimeError('Source suite must have completed successfully')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'suite.json'
    if manifest.exists():raise FileExistsError(manifest)
    status=dict(pid=os.getpid(),started=time.time(),source=str(args.source),runs=[])
    manifest.write_text(json.dumps(status,indent=2))
    for source_run in original['runs']:
        name=source_run['name']
        command=[sys.executable,str(Path(__file__).with_name('run.py')),'train',
                 '--resume-from',str(args.source/name/'last.pt'),
                 '--additional-steps',str(args.additional_steps),'--output',str(args.output/name)]
        if args.allow_resume_code_change:command.append('--allow-resume-code-change')
        row=dict(name=name,command=command,started=time.time(),status='running')
        status['runs'].append(row)
        with (args.output/f'{name}.log').open('w') as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=proc.pid;manifest.write_text(json.dumps(status,indent=2))
            result=proc.wait()
        row.update(exit_code=result,finished=time.time(),status='completed' if result==0 else 'failed')
        manifest.write_text(json.dumps(status,indent=2));print(name,row['status'],flush=True)
    status['finished']=time.time();manifest.write_text(json.dumps(status,indent=2))
    if any(r['exit_code'] for r in status['runs']):raise SystemExit(1)


if __name__=='__main__':main()
