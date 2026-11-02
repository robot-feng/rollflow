#!/usr/bin/env python
"""交互式选择或一键生成 RollFlow 迁移配置；不自动重写任何策略代码。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assets"))
from configuration import Selection, validate_resolved


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interactive", action="store_true", help="逐项选择；回车采用默认值")
    p.add_argument("--yes", action="store_true", help="直接采用默认/明确指定值，不询问")
    p.add_argument("--preset", choices=["quick", "lsd-original"], default="quick")
    p.add_argument("--policy-name", default="native_policy")
    p.add_argument("--horizon", type=int)
    p.add_argument("--execution-chunks", type=int, help="等长执行块数；默认 4")
    p.add_argument("--actions-per-output", type=int, help="每轮输出 C；默认 H/4")
    p.add_argument("--initialization", choices=["native", "finetune", "scratch"], default="native")
    p.add_argument("--checkpoint-horizon", type=int, help="已核对的原策略窗口；默认保留它")
    p.add_argument("--allow-horizon-change", action="store_true", help="显式允许提出修改旧窗口；仍需专项迁移验证")
    p.add_argument("--group-domain", choices=["actions", "execution_chunks"])
    p.add_argument("--single-group-probability", type=float)
    p.add_argument("--native-time-convention", choices=["from_native", "noise_at_zero", "noise_at_one"], default="from_native")
    p.add_argument("--original-recipe", choices=["core", "starvla-wrapper"], default="core")
    p.add_argument("--refinements", type=int, help="LSD 每动作累计细化次数 R；instant 固定 H/C")
    p.add_argument("--cold-steps", type=int, help="原始 LSD 冷启动 1 次或 R 次；instant 固定 H/C")
    p.add_argument("--output", type=Path, default=Path("rollflow_config.json"))
    return p


def prompt(label: str, default: str, input_fn=input) -> str:
    answer = input_fn(f"{label} [{default}]: ").strip()
    return answer or default


def select_interactively(a, input_fn=input) -> None:
    choice = prompt("路线：1 快速尝鲜（瞬时FM）/ 2 原始LSD / 3 自定义瞬时FM", "2" if a.preset == "lsd-original" else "1", input_fn)
    if choice not in {"1", "2", "3"}:
        raise ValueError("路线请输入 1、2 或 3")
    a.preset = "lsd-original" if choice == "2" else "quick"
    a.horizon = int(prompt("预测窗口 H", str(a.horizon or a.checkpoint_horizon or 64), input_fn))
    if a.actions_per_output is not None and a.execution_chunks is None:
        # 已明确 C，不重复询问不必要的派生数。
        a.actions_per_output = int(prompt("每轮输出并执行的动作数 C", str(a.actions_per_output), input_fn))
    else:
        a.execution_chunks = int(prompt("执行块数 M（不是训练噪声分组数）", str(a.execution_chunks or 4), input_fn))
    if choice == "3":
        a.group_domain = prompt("训练分组粒度 actions / execution_chunks", a.group_domain or "actions", input_fn)
    if choice == "2":
        print("LSD 需要区间条件和额外训练计算；一步/两步效果必须另测。")
        a.original_recipe = prompt("原始配方 core / starvla-wrapper", a.original_recipe, input_fn)
        m = a.execution_chunks or a.horizon // a.actions_per_output
        a.refinements = int(prompt("每动作累计细化 R", str(a.refinements or m), input_fn))
        a.cold_steps = int(prompt("冷启动调用数 N（1 或 R）", str(a.cold_steps or 1), input_fn))


def main(argv=None) -> int:
    p = parser()
    a = p.parse_args(argv)
    if a.interactive and a.yes:
        p.error("--interactive 与 --yes 不能同时使用")
    try:
        if a.interactive:
            select_interactively(a)
        selected = Selection(**{key: getattr(a, key) for key in Selection.__dataclass_fields__})
        config = selected.resolve()
        validate_resolved(config)
        payload = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        a.output.parent.mkdir(parents=True, exist_ok=True)
        # 拒绝覆盖已有配置；更改配置使用新路径，便于比较和回滚。
        with a.output.open("x", encoding="utf-8") as f:
            f.write(payload)
    except (ValueError, FileExistsError, EOFError, OSError) as error:
        p.error(str(error))
    g, inf = config["geometry"], config["inference"]
    print(f"已创建 {a.output}")
    print(f"独立算法：{config['registered_policy']}")
    print(f"H={g['horizon']}，执行块数={g['execution_chunks']}，每轮输出={g['actions_per_output']}")
    print(f"冷启动专家前向={inf['cold_nfe']}，稳态每轮={inf['warm_nfe']}，每动作累计细化={inf['lifecycle_refinements']}")
    print(f"训练噪声分组候选={config['training']['group_choices']}")
    print("状态：仅创建迁移配置；尚未修改或验证目标模型。")
    for message in config["warnings"]:
        print("注意：" + message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
