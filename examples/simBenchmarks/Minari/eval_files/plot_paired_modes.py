"""Plot rolling K comparisons relative to a per-seed paired reference.

Each JSON must contain per-mode returns and explicit environment seeds. The
confidence bands resample environment seeds within one training seed; they do
not estimate variability across trained models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

comparison_path = Path(__file__).resolve().parents[1] / 'compare_few_step.py'
spec = importlib.util.spec_from_file_location('compare_few_step', comparison_path)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def compute_panel(reference, candidates):
    """Return per-K paired effects; reject mismatched environment IDs."""
    reference_entry = reference['instant_k4']
    seeds = comparison.environment_seeds(reference_entry, allow_legacy=False)
    ref_returns = reference_entry['environment_returns']
    result = {}
    for label, entries, mode in candidates:
        rows = []
        for k in (1, 2, 4, 8):
            entry = entries[f'{mode}_k{k}']
            if comparison.environment_seeds(entry, allow_legacy=False) != seeds:
                raise ValueError(f'Environment seeds differ for {label}, K={k}')
            difference = comparison.bootstrap_difference(entry['environment_returns'], ref_returns)
            rows.append({'k': k, 'mean_return': float(entry['mean_return']), **difference})
        result[label] = rows
    return {'reference_mean_return': float(reference_entry['mean_return']),
            'environment_seeds': seeds, 'curves': result}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in (42, 43):
        for kind in ('fm', 'uniform', 'biased'):
            parser.add_argument(f'--seed{seed}-{kind}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panels, sources = {}, {}
    for seed in (42, 43):
        paths = {kind: getattr(args, f'seed{seed}_{kind}') for kind in ('fm', 'uniform', 'biased')}
        data = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
        panels[str(seed)] = compute_panel(data['fm'], [
            ('FM30 instant', data['fm'], 'instant'),
            ('FM+LSD+exec60 uniform', data['uniform'], 'average'),
            ('FM+LSD+exec60 biased', data['biased'], 'average'),
        ])
        sources[str(seed)] = {kind: {'path': str(path), 'sha256': sha(path)} for kind, path in paths.items()}

    args.output.mkdir(parents=True)
    result = {'reference': 'each training seed FM30 instant K=4',
              'uncertainty': 'paired bootstrap over environment seeds within each training seed',
              'sources': sources, 'panels': panels}
    (args.output / 'paired_curves.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    colors = ('#555555', '#2478a5', '#dc7b24')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, seed in zip(axes, ('42', '43')):
        panel = panels[seed]
        for color, (label, rows) in zip(colors, panel['curves'].items()):
            x = np.arange(len(rows))
            means = np.array([row['mean'] for row in rows])
            lower = np.array([row['ci95'][0] for row in rows])
            upper = np.array([row['ci95'][1] for row in rows])
            axis.plot(x, means, 'o-', color=color, label=label)
            axis.fill_between(x, lower, upper, color=color, alpha=0.12)
        axis.axhline(0, color='black', linewidth=0.8)
        axis.set_xticks(range(4), ('1', '2', '4', '8'))
        axis.set_xlabel('Rolling refinement K')
        axis.set_title(f"Training seed {seed}; FM30 K4 = {panel['reference_mean_return']:.2f}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('Paired return difference vs FM30 instant K4')
    axes[1].legend(fontsize=8)
    fig.suptitle('Pusher rolling comparison; shaded: paired environment bootstrap 95% CI')
    fig.tight_layout()
    fig.savefig(args.output / 'paired_curves.png', dpi=180)
    plt.close(fig)
    print(args.output / 'paired_curves.png')


if __name__ == '__main__':
    main()
