"""Continue an LSD checkpoint while changing only its LSD clock distribution.

FM always uses the original sampled clocks, noise and targets. Both experiment
arms split the FM and LSD forwards identically; only the near-deployment arm
replaces half of non-diagonal LSD rows. This is a process-local experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.action_model.rolling_meanflow_matching_head import rolling_meanflow as core

_original_objective = core.central_difference_lsd


def lsd_times(s, t, delta, chunk_size, mode):
    """Return independent LSD clocks without changing inputs or consuming RNG."""
    if mode not in ("original", "near_deployment"):
        raise ValueError("Unknown LSD clock mode")
    selected = torch.zeros(s.shape[0], 1, 1, device=s.device, dtype=torch.bool)
    if mode == "original":
        return s, t, selected
    selected = ((torch.arange(s.shape[0], device=s.device) % 2 == 0)[:, None, None]
                & (t > s).flatten(1).any(1)[:, None, None])
    # Production training selects one group count for the full batch; its
    # sorted targets are repeated across equal blocks. Derive that count from
    # the unmodified first row before any padding cleanup.
    groups = 1 + int((t[0, 1:, 0] != t[0, :-1, 0]).sum())
    if t.shape[1] % groups:
        raise ValueError("Expected equally sized original training time groups")
    sampler = core.StaircaseTimeSampler(core.RollFlowConfig(
        horizon=t.shape[1], action_dim=1, chunk_size=chunk_size,
        finite_difference_delta=delta,
    ))
    if not sampler.cfg.valid_steps(groups):
        raise ValueError("Original group count exceeds deployment refinement levels")
    lower = s.new_zeros(s.shape[0])
    upper = s.new_full((s.shape[0],), 1.0 - delta - 1e-6)
    near_s, near_t, _ = sampler._build(lower, upper, groups)
    return torch.where(selected, near_s, s), torch.where(selected, near_t, t), selected


def split_objective(model, x0, x1, s, t, *, chunk_size, mode, delta,
                    active=None, context=None, pad=None, w_fm=1.0, w_lsd=0.5,
                    lsd_loss_space="velocity", model_kwargs=None):
    """Keep the original FM term even when changing the separate LSD clocks."""
    common = dict(delta=delta, context=context, pad=pad, model_kwargs=model_kwargs,
                  lsd_loss_space=lsd_loss_space)
    fm_loss, fm_stats = _original_objective(
        model, x0, x1, s, t, active=active, w_fm=w_fm, w_lsd=0.0, **common
    )
    if w_lsd <= 0:
        return fm_loss, fm_stats
    lsd_s, lsd_t, selected = lsd_times(s, t, delta, chunk_size, mode)
    lsd_active = (active if mode == "original" and active is not None
                  else core._lsd_mask(lsd_s, lsd_t, delta))
    lsd_loss, lsd_stats = _original_objective(
        model, x0, x1, lsd_s, lsd_t, active=lsd_active,
        w_fm=0.0, w_lsd=w_lsd, **common
    )
    stats = dict(fm_stats)
    for key in ("lsd_loss", "active_lsd_frac", "v_tangent_abs", "v_teacher_abs"):
        stats[key] = lsd_stats[key]
    stats["lsd_replaced_row_frac"] = selected.float().mean().detach()
    stats["lsd_source_time_min"] = lsd_s.min().detach()
    stats["lsd_source_time_max"] = lsd_s.max().detach()
    stats["lsd_terminal_time_min"] = lsd_t.min().detach()
    stats["lsd_terminal_time_max"] = lsd_t.max().detach()
    return fm_loss + lsd_loss, stats


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lsd-clock-mode", choices=("original", "near_deployment"), required=True)
    options, remaining = parser.parse_known_args(sys.argv[1:])
    if not remaining or remaining[0] != "train" or "--resume-from" not in remaining or "--output" not in remaining:
        raise SystemExit("Requires train --resume-from CHECKPOINT --output OUTPUT")
    resume_path = Path(remaining[remaining.index("--resume-from") + 1])
    output = Path(remaining[remaining.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if checkpoint["arguments"]["variant"] != "rolling_lsd":
        raise ValueError("Requires a rolling_lsd checkpoint")
    previous = Path(checkpoint["arguments"]["output"])
    # An endpoint or sampling ablation cannot be silently removed by this entry.
    for marker in ("endpoint_consistency_ablation.json", "deployment_clock_ablation.json"):
        if (previous / marker).is_file():
            raise ValueError(f"Start from ordinary FM+LSD, found {previous / marker}")
    metadata = previous / "lsd_clock_only_ablation.json"
    if metadata.is_file() and json.loads(metadata.read_text())["mode"] != options.lsd_clock_mode:
        raise ValueError("Cannot change LSD clock mode across an ordinary resume")
    chunk_size = int(checkpoint["arguments"]["chunk"])
    del checkpoint
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    (output / "lsd_clock_only_ablation.json").write_text(json.dumps({
        "mode": options.lsd_clock_mode,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resumed_from": str(resume_path),
        "fm": "original sampled times, noise, targets and weight",
        "lsd": "same central difference objective; optionally replace even non-diagonal rows",
        "changed_fraction": "half by batch index, intersected with non-diagonal rows",
        "group_count": "same as the original sampled batch",
        "near_upper": "1-delta-1e-6", "near_lower": 0.0,
        "forward_path": "same split FM/LSD computation in both clock modes",
        "clock_diagnostics": "s/t/source_ratio remain the original FM clocks; lsd_* records the replacement",
        "production_model_unchanged": True,
    }, indent=2) + "\n")

    diagnostics = {}
    def objective(model, x0, x1, s, t, **kwargs):
        loss, stats = split_objective(model, x0, x1, s, t, chunk_size=chunk_size,
                                     mode=options.lsd_clock_mode, **kwargs)
        for key in ("lsd_replaced_row_frac", "lsd_source_time_min", "lsd_source_time_max",
                    "lsd_terminal_time_min", "lsd_terminal_time_max"):
            if key in stats:
                diagnostics[key] = float(stats[key].cpu())
        return loss, stats

    original_rollflow_loss = core.RollFlow.loss
    def rollflow_loss(self, *args, **kwargs):
        diagnostics.clear()
        loss, stats = original_rollflow_loss(self, *args, **kwargs)
        stats.update(diagnostics)
        return loss, stats

    core.central_difference_lsd = objective
    core.RollFlow.loss = rollflow_loss
    os.environ["STARVLA_MINARI_INTERVENTION"] = "lsd_clock_only"
    sys.argv = sys.argv[:1] + remaining
    from examples.simBenchmarks.Minari import run
    run.main()


if __name__ == "__main__":
    main()
