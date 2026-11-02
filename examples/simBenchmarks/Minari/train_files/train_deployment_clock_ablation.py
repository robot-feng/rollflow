"""Experimental clock intervention for continuing a Minari LSD checkpoint.

Half of off-diagonal rows use randomized, deployment-shaped staircase clocks.
Other rows and all diagonal FM rows retain the production sampler. This wrapper
only patches this process; it does not change the production RollFlow sampler.
Invoke with the same arguments as run.py, including train --resume-from.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    StaircaseTimeSampler, _lsd_mask,
)

_original = StaircaseTimeSampler.sample_training


def sample_training(self, batch, *, device, dtype=None, generator=None, p_fm=None,
                    num_time_groups=None):
    import torch
    if dtype is None:
        dtype = torch.float32
    times = _original(self, batch, device=device, dtype=dtype, generator=generator,
                      p_fm=p_fm, num_time_groups=num_time_groups)
    # During pure FM warmup, preserve the original output and RNG exactly.
    if bool(times.fm_mask.all()):
        return times
    device = torch.device(device)
    selected = (torch.rand(batch, 1, 1, device=device, generator=generator) < .5) & ~times.fm_mask
    levels = [k for k in (1, 2, 4, 8) if self.cfg.valid_steps(k)]
    k = levels[int(torch.randint(len(levels), (1,), device=device, generator=generator).item())]
    # Keep t < 1-delta so centered LSD includes the first, executed chunk.
    upper_max = 1.0 - self.cfg.finite_difference_delta - .005
    if upper_max <= .8:
        raise ValueError('delta too large for this ablation')
    upper = .8 + (upper_max - .8) * torch.rand(batch, device=device, dtype=dtype, generator=generator)
    lower = .1 * torch.rand(batch, device=device, dtype=dtype, generator=generator)
    s_deploy, t_deploy, _ = self._build(lower, upper, k)
    s = torch.where(selected, s_deploy, times.s)
    t = torch.where(selected, t_deploy, times.t)
    # No single ratio exists for deployment-shaped clocks. Store its per-row
    # mean only for the existing source_ratio diagnostic.
    mean_ratio = torch.where(t > 0, s / t.clamp_min(1e-8), torch.ones_like(t)).mean(dim=1, keepdim=True)
    ratio = torch.where(selected, mean_ratio, times.ratio)
    return replace(times, s=s, t=t, active=_lsd_mask(s, t, self.cfg.finite_difference_delta), ratio=ratio)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'train' or '--resume-from' not in sys.argv or '--output' not in sys.argv:
        raise SystemExit('Requires train --resume-from CHECKPOINT --output OUTPUT (run.py arguments)')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    destination = output / 'deployment_clock_ablation.py'
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    (output / 'deployment_clock_ablation.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'intervention': 'half of off-diagonal rows use randomized deployment-shaped clocks',
        'upper_range': '[0.8, 1-delta-0.005]', 'lower_range': '[0, 0.1]',
        'levels': [1, 2, 4, 8], 'all_fm_rows_original': True,
        'source_ratio_metric': 'mean per-token s/t for changed rows',
        'num_time_groups_metric': 'original sampler value, not intervention level',
    }, indent=2) + '\n')
    StaircaseTimeSampler.sample_training = sample_training
    os.environ['STARVLA_MINARI_INTERVENTION'] = 'clock'
    from examples.simBenchmarks.Minari import run
    run.main()


if __name__ == '__main__':
    main()
