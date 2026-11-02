#!/usr/bin/env python
"""Verify package content hashes (integrity, not an authenticity signature)."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/'BUNDLE_SHA256.json').read_text(encoding='utf8'))
    for name, expected in manifest['files'].items():
        p=(root/name).resolve()
        if root not in p.parents:
            raise ValueError('Manifest path escapes the bundle')
        if not p.is_file() or sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or altered bundle file: '+name)
    print(f"Bundle integrity: {len(manifest['files'])} files match")


if __name__=='__main__':verify()
