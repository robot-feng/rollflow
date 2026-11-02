#!/usr/bin/env python
"""Run local core tests, provenance checks, and the quick-start example.

Does not install dependencies, edit target repositories, or run external VLA checkpoints.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-example',action='store_true')
    a=p.parse_args()
    commands=[]
    if (ROOT/'BUNDLE_SHA256.json').exists():
        commands.append([sys.executable,'scripts/verify_bundle.py'])
    commands += [[sys.executable,'tools/check_originals.py'],[sys.executable,'-m','pytest','-q','tests']]
    if not a.skip_example:
        commands.append([sys.executable,'examples/quickstart.py','--steps','24'])
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    for command in commands:
        print('\nRUN: '+' '.join(command),flush=True)
        result=subprocess.run(command,cwd=ROOT,env=env,check=False)
        if result.returncode:
            return result.returncode
    print('\nLocal checks passed; production-policy validation remains target-specific.')
    return 0


if __name__=='__main__':raise SystemExit(main())
