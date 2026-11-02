"""Sequential three-arm experiment; suitable for running inside tmux.

Use distinct output roots for separate suites. Failed arms are recorded and
the remaining comparisons continue; existing arms are never overwritten.
"""
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
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--precisions',nargs='+',choices=['fp32','bf16'],default=['fp32','bf16'])
    parser.add_argument('--variants',nargs='+',choices=['fm','rolling_fm','rolling_lsd'],
                        default=['fm','rolling_fm','rolling_lsd'],
                        help='Choose arms; rolling_lsd before warmup is the same head in FM-only mode')
    args,extra=parser.parse_known_args()
    args.output.mkdir(parents=True,exist_ok=True)
    status_path=args.output/'suite.json'
    if status_path.exists():raise FileExistsError('Use a new suite output root')
    status={'pid':os.getpid(),'started':time.time(),'runs':[]}
    status_path.write_text(json.dumps(status,indent=2))
    for seed in args.seeds:
        for precision in args.precisions:
            for variant in args.variants:
                name=f'{variant}_{precision}_seed{seed}'
                command=[sys.executable,str(Path(__file__).with_name('run.py')),'train',
                         *extra,'--variant',variant,'--precision',precision,
                         '--seed',str(seed),'--output',str(args.output/name)]
                record=dict(name=name,command=command,started=time.time(),status='running')
                status['runs'].append(record)
                with (args.output/f'{name}.log').open('w') as log:
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    record['pid']=proc.pid
                    status_path.write_text(json.dumps(status,indent=2))
                    result=proc.wait()
                record.update(exit_code=result,finished=time.time(),status='completed' if result==0 else 'failed')
                status_path.write_text(json.dumps(status,indent=2))
                print(name,record['status'],flush=True)
    status['finished']=time.time()
    status_path.write_text(json.dumps(status,indent=2))
    if any(r['exit_code'] for r in status['runs']):raise SystemExit(1)


if __name__=='__main__':main()
