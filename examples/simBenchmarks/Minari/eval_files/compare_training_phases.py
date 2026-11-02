"""Compare instantaneous FM and distilled mean policies from one training lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_few_step import bootstrap_difference, environment_seeds, paired_episodes


def compare(reference, candidates, steps, candidate_velocity="average", reference_velocity="instant"):
    result = []
    for label, evaluation in candidates:
        for k in steps:
            base = reference[f"{reference_velocity}_k{k}"]
            candidate = evaluation[f"{candidate_velocity}_k{k}"]
            if environment_seeds(base, False) != environment_seeds(candidate, False):
                raise ValueError("Environment seeds differ")
            candidate_mse, base_mse = paired_episodes(candidate["replay_episodes"], base["replay_episodes"])
            result.append({
                "candidate": label, "k": k,
                "reference_return": base["mean_return"], "candidate_return": candidate["mean_return"],
                "reference_replay_mse": base["replay_normalized_mse"],
                "candidate_replay_mse": candidate["replay_normalized_mse"],
                "reference_mean_dit_calls": float(np.mean(base["dit_calls_per_chunk"])),
                "candidate_mean_dit_calls": float(np.mean(candidate["dit_calls_per_chunk"])),
                "reference_max_dit_calls": max(base["dit_calls_per_chunk"]),
                "candidate_max_dit_calls": max(candidate["dit_calls_per_chunk"]),
                "return_difference": bootstrap_difference(candidate["environment_returns"], base["environment_returns"]),
                "episode_mse_difference": bootstrap_difference(candidate_mse, base_mse),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="FM-only checkpoint evaluation")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="Distilled checkpoint evaluations")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--candidate-velocity", choices=["average", "instant"], default="average")
    parser.add_argument("--reference-velocity", choices=["average", "instant"], default="instant")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.k or len(set(args.k)) != len(args.k) or any(k <= 0 for k in args.k):
        parser.error("k must contain distinct positive values")
    base = json.loads(args.reference.read_text())
    candidates = [(str(path), json.loads(path.read_text())) for path in args.candidates]
    result = compare(base, candidates, args.k, args.candidate_velocity, args.reference_velocity)
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps({"reference": str(args.reference),
        "reference_velocity": args.reference_velocity, "candidate_velocity": args.candidate_velocity,
        "results": result}, indent=2, allow_nan=False) + "\n")
    candidate_mode = "均值" if args.candidate_velocity == "average" else "瞬时"
    reference_mode = "均值" if args.reference_velocity == "average" else "瞬时"
    report = ["# 相同 K 的策略配对比较", "",
              f"参考：`{args.reference}`，使用{reference_mode}速度；候选使用{candidate_mode}速度。",
              "候选减参考，return 越高越好、MSE 越低越好；区间按相同环境种子配对 bootstrap。",
              "比较同一模型训练前后时须核对 checkpoint 血缘；本报告不衡量训练种子方差。", "",
              "平均 DiT 次数混合了冷启动和热态，最大次数为记录中单个 chunk 的最大值；它们不是延迟测量。", "",
              f"|候选 checkpoint|K|参考：{reference_mode} return|候选：{candidate_mode} return|return 差及 95% 区间|参考/候选回放 MSE|参考/候选平均 DiT 次数|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result:
        diff = row["return_difference"]
        report.append(f"|{row['candidate']}|{row['k']}|{row['reference_return']:.2f}|{row['candidate_return']:.2f}|"
                      f"{diff['mean']:.2f} [{diff['ci95'][0]:.2f}, {diff['ci95'][1]:.2f}]|"
                      f"{row['reference_replay_mse']:.5f} / {row['candidate_replay_mse']:.5f}|"
                      f"{row['reference_mean_dit_calls']:.3f} / {row['candidate_mean_dit_calls']:.3f}|")
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
