"""Branch a pure FM continuation from a rolling LSD warmup checkpoint.

Model, optimizer, scheduler, RNG and data split are retained. The default
forces diagonal sampling. Matched sampling keeps the source p_fm schedule to
isolate LSD while the FM network still receives (t,t) in both cases.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def branch_payload(payload, *, match_time_sampling=False):
    args, config = payload['arguments'], payload['head_config']
    if args['variant'] != 'rolling_lsd':
        raise ValueError('Expected a rolling_lsd checkpoint')
    if payload['step'] != args['fm_warmup']:
        raise ValueError('Branch only at the FM warmup boundary')
    for key in ('w_lsd', 'p_fm'):
        if float(args[key]) != float(config[key]):
            raise ValueError(f'Checkpoint {key} arguments and head config disagree')
    for key in ('model', 'optimizer', 'scheduler', 'split', 'normalization', 'torch_rng'):
        if key not in payload:
            raise ValueError(f'Checkpoint is missing {key}')
    old = {key: float(args[key]) for key in ('w_lsd', 'p_fm')}
    old['variant'] = args['variant']
    old['velocity_mode'] = config['velocity_mode']
    args['w_lsd'] = config['w_lsd'] = 0.0
    if not match_time_sampling:
        args['p_fm'] = config['p_fm'] = 1.0
    args['variant'] = 'rolling_fm'
    config['velocity_mode'] = 'instant'
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-time-sampling', action='store_true',
                        help='Keep source p_fm schedule to match LSD FM-time sampling')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old = branch_payload(payload, match_time_sampling=args.match_time_sampling)
    source_manifest = Path(payload['arguments']['output']) / 'manifest.json'
    if not source_manifest.is_file():
        raise FileNotFoundError(f'Source run manifest required for resume: {source_manifest}')
    args.output.mkdir(parents=True)
    destination = args.output / 'branch.pt'
    temporary = args.output / 'branch.pt.tmp'
    torch.save(payload, temporary)
    temporary.replace(destination)
    provenance = {'source_checkpoint': str(args.checkpoint.resolve()),
                  'source_sha256': digest(args.checkpoint),
                  'branch_checkpoint': str(destination.resolve()),
                  'branch_sha256': digest(destination), 'step': payload['step'],
                  'source_values': old, 'branch_values': {'w_lsd': 0., 'p_fm': payload['arguments']['p_fm'],
                                                         'variant': 'rolling_fm', 'velocity_mode': 'instant'},
                  'changed_fields': ['arguments.w_lsd', 'head_config.w_lsd',
                                     'arguments.variant', 'head_config.velocity_mode'] +
                                    ([] if args.match_time_sampling else ['arguments.p_fm', 'head_config.p_fm']),
                  'match_time_sampling': args.match_time_sampling,
                  'note': 'Model, optimizer, scheduler, RNG and data split are copied.'}
    (args.output / 'branch_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
