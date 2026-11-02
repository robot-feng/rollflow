"""Plot matched Minari checkpoint evaluations from eval_checkpoints.sh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_progress(root: Path):
    records = []
    for directory in root.glob("step_*"):
        if not directory.is_dir() or not directory.name[5:].isdigit():
            continue
        path = directory / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        evaluation = json.loads(path.read_text())
        records.append((int(directory.name[5:]), evaluation))
    records.sort(key=lambda item: item[0])
    if not records:
        raise ValueError(f"No checkpoint evaluations in {root}")
    modes = set(records[0][1])
    seeds = None
    replay_ids = None
    summary = []
    for step, evaluation in records:
        if set(evaluation) != modes:
            raise ValueError("Evaluation modes differ between checkpoints")
        metrics = {}
        for mode in sorted(modes):
            value = evaluation[mode]
            mode_seeds = value.get("environment_seeds")
            if mode_seeds is None or len(mode_seeds) != len(value["environment_returns"]):
                raise ValueError("Each evaluation must record its environment seeds")
            if len(set(mode_seeds)) != len(mode_seeds):
                raise ValueError("Duplicate environment seeds")
            if seeds is None:
                seeds = mode_seeds
            elif mode_seeds != seeds:
                raise ValueError("Environment seeds differ between modes or checkpoints")
            ids = [row["episode_id"] for row in value["replay_episodes"]]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate replay episode IDs")
            if replay_ids is None:
                replay_ids = ids
            elif ids != replay_ids:
                raise ValueError("Replay episode IDs differ between modes or checkpoints")
            metrics[mode] = {
                "mean_return": value["mean_return"],
                "replay_normalized_mse": value["replay_normalized_mse"],
                "average_dit_calls": float(np.mean(value["dit_calls_per_chunk"])),
            }
        summary.append({"step": step, "metrics": metrics})
    return {"environment_seeds": seeds, "replay_episode_ids": replay_ids, "checkpoints": summary}


def merge_progress(roots):
    """Combine a run and its continuation after checking evaluation alignment."""
    inputs = [read_progress(root) for root in roots]
    result = {"environment_seeds": inputs[0]["environment_seeds"],
              "replay_episode_ids": inputs[0]["replay_episode_ids"], "checkpoints": []}
    modes = None
    seen_steps = set()
    for data in inputs:
        if data["environment_seeds"] != result["environment_seeds"]:
            raise ValueError("Environment seeds differ between input directories")
        if data["replay_episode_ids"] != result["replay_episode_ids"]:
            raise ValueError("Replay episodes differ between input directories")
        for item in data["checkpoints"]:
            if item["step"] in seen_steps:
                raise ValueError("Duplicate checkpoint steps between input directories")
            seen_steps.add(item["step"])
            current_modes = set(item["metrics"])
            if modes is None:
                modes = current_modes
            elif current_modes != modes:
                raise ValueError("Evaluation modes differ between input directories")
            result["checkpoints"].append(item)
    result["checkpoints"].sort(key=lambda item: item["step"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more output directories from eval_checkpoints.sh")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = merge_progress(args.input)
    checkpoints = result["checkpoints"]
    steps = [item["step"] for item in checkpoints]
    modes = list(checkpoints[0]["metrics"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in modes:
        returns = [item["metrics"][mode]["mean_return"] for item in checkpoints]
        mse = [item["metrics"][mode]["replay_normalized_mse"] for item in checkpoints]
        axes[0].plot(steps, returns, marker="o", label=mode)
        axes[1].plot(steps, mse, marker="o", label=mode)
    axes[0].set_ylabel("Mean environment return (higher is better)")
    axes[1].set_ylabel("Replay normalized MSE (lower is better)")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "progress.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    input_labels = ", ".join(f"`{path}`" for path in args.input)
    rows = ["# Checkpoint progress", "",
        f"Inputs: {input_labels}. Each checkpoint uses {len(result['environment_seeds'])} matched environment seeds.",
        "Checkpoints must come from the same run or its continuation; this chart does not measure variation across training seeds.", "",
        "|Step|Mode|Mean return|Replay normalized MSE|Mean DiT calls/chunk|", "|---:|---|---:|---:|---:|"]
    for checkpoint in checkpoints:
        for mode, value in checkpoint["metrics"].items():
            rows.append(f"|{checkpoint['step']}|{mode}|{value['mean_return']:.3f}|"
                        f"{value['replay_normalized_mse']:.5f}|{value['average_dit_calls']:.3f}|")
    rows.extend(["", "![Checkpoint progress](progress.png)", ""])
    (args.output / "report.md").write_text("\n".join(rows))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
