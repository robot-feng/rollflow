"""创建迁移配置，不导入、不改写目标仓库，也不声称自动完成适配。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import json
import re
from typing import Any

SCHEMA_VERSION = "rollflow-migration/1.0"


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须是正整数，不能是浮点数或布尔值")
    return value


def divisors(value: int) -> list[int]:
    positive_int(value, "value")
    return [g for g in range(1, value + 1) if value % g == 0]


def probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是 [0,1] 的有限数值")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} 必须是 [0,1] 的有限数值")
    return float(value)


@dataclass(frozen=True)
class Selection:
    """None 表示尚未指定，由默认值或已知 checkpoint 元数据解析。"""
    preset: str = "quick"
    policy_name: str = "native_policy"
    horizon: int | None = None
    execution_chunks: int | None = None
    actions_per_output: int | None = None
    initialization: str = "native"
    checkpoint_horizon: int | None = None
    allow_horizon_change: bool = False
    group_domain: str | None = None
    single_group_probability: float | None = None
    native_time_convention: str = "from_native"
    original_recipe: str = "core"
    refinements: int | None = None
    cold_steps: int | None = None

    def resolve(self) -> dict[str, Any]:
        if self.preset not in {"quick", "lsd-original"}:
            raise ValueError("preset 必须是 quick 或 lsd-original；自定义通过参数覆盖")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.policy_name):
            raise ValueError("policy_name 必须是小写字母开头的字母/数字/下划线标识符")
        if self.initialization not in {"native", "finetune", "scratch"}:
            raise ValueError("initialization 必须是 native、finetune 或 scratch")
        if type(self.allow_horizon_change) is not bool:
            raise ValueError("allow_horizon_change 必须是布尔值")
        if self.native_time_convention not in {"from_native", "noise_at_zero", "noise_at_one"}:
            raise ValueError("必须审计并保留原生时间约定，不能随意翻转")
        if self.original_recipe not in {"core", "starvla-wrapper"}:
            raise ValueError("original_recipe 必须明确选择 core 或 starvla-wrapper")
        if self.checkpoint_horizon is not None:
            positive_int(self.checkpoint_horizon, "checkpoint_horizon")
        if self.initialization == "finetune" and self.checkpoint_horizon is None:
            raise ValueError("finetune 请先核对 checkpoint_horizon；不得猜测旧权重长度")
        if self.initialization == "scratch" and self.checkpoint_horizon is not None:
            raise ValueError("scratch 与已有策略 checkpoint_horizon 冲突；骨干预训练另由原工程管理")

        h = self.horizon if self.horizon is not None else (self.checkpoint_horizon or 64)
        positive_int(h, "horizon")
        warnings: list[str] = []
        if self.checkpoint_horizon is not None and h != self.checkpoint_horizon:
            if not self.allow_horizon_change:
                raise ValueError("所选 horizon 与旧权重不一致；默认保留旧长度，禁止静默插值/扩展位置参数")
            warnings.append("已显式要求改变 horizon；必须单独验证数据窗口、位置参数及 checkpoint 迁移")
        m, c = self.execution_chunks, self.actions_per_output
        if m is None and c is None:
            m = 4
        if m is not None:
            positive_int(m, "execution_chunks")
            if h % m:
                raise ValueError(f"H={h} 不能分为 {m} 个等长执行块；可选块数：{divisors(h)}")
        if c is not None:
            positive_int(c, "actions_per_output")
            if h % c:
                raise ValueError(f"每次输出 C={c} 必须整除 H={h}")
        if m is None:
            m = h // c
        if c is None:
            c = h // m
        if m * c != h:
            raise ValueError("配置冲突：horizon 必须等于 execution_chunks × actions_per_output")

        is_lsd = self.preset == "lsd-original"
        domain = self.group_domain or ("execution_chunks" if is_lsd else "actions")
        if domain not in {"actions", "execution_chunks"}:
            raise ValueError("group_domain 必须是 actions 或 execution_chunks")
        if is_lsd and domain != "execution_chunks":
            raise ValueError("原始 LSD 的候选必须为 Div(H/C)，不能静默换成 Div(H)")
        choices = divisors(h if domain == "actions" else m)
        p1 = self.single_group_probability
        if p1 is None:
            p1 = 0.7 if is_lsd else 1 / len(choices)
        p1 = probability(p1, "single_group_probability")
        weights = [1.0] if len(choices) == 1 else [p1] + [(1-p1)/(len(choices)-1)]*(len(choices)-1)
        r = self.refinements if self.refinements is not None else m
        n = self.cold_steps if self.cold_steps is not None else (1 if is_lsd else m)
        positive_int(r, "refinements")
        positive_int(n, "cold_steps")
        if r > m:
            raise ValueError("refinements 不能超过执行块数 H/C")
        if not is_lsd and (r != m or n != m):
            raise ValueError("quick instant 内核要求 R=N=H/C；任意缩短步数需另选并验证算法")
        if is_lsd and n not in {1, r}:
            raise ValueError("原始执行器只支持一次冷启动或 R 次迭代冷启动，不支持任意 N")

        warmup = 20_000 if self.original_recipe == "core" else 5_000
        p_fm = 0.7 if self.original_recipe == "core" else 0.3
        if is_lsd:
            warnings.extend([
                "LSD 是显式可选路线：需要区间条件与额外前向/梯度；少步质量没有自动保证",
                "original_absolute_fd_v1 保留原始 active-token 的绝对 t±δ；不能冒称为直线路径 LSD",
                "原始 ratio 采样与 warm 阶梯不完全匹配，耦合 token 的路径一致性仍需宿主专项验证",
            ])
        if self.checkpoint_horizon is not None and self.horizon is None and h != 64:
            warnings.append(f"保留旧策略 H={h}，未套用新初始化默认 H=64")
        algorithm = f"{self.policy_name}_rollflow_{'lsd' if is_lsd else 'instant'}"
        return {
            "schema_version": SCHEMA_VERSION,
            "preset": self.preset,
            "baseline_policy": self.policy_name,
            "registered_policy": algorithm,
            "initialization": self.initialization,
            "checkpoint_horizon": self.checkpoint_horizon,
            "allow_horizon_change": self.allow_horizon_change,
            "geometry": {"horizon": h, "execution_chunks": m, "actions_per_output": c},
            "model": {
                "velocity_mode": "average" if is_lsd else "instant",
                "time_shape": "[B,H]",
                "native_time_convention": self.native_time_convention,
                "new_trainable_parameters": "endpoint_branch_only" if is_lsd else "none_expected",
                "parameter_loading": "explicit_new_keys_only" if is_lsd else "strict",
            },
            "training": {
                "group_domain": domain,
                "group_unit": c if domain == "execution_chunks" else 1,
                "group_choices": choices,
                "group_probabilities": weights,
                "single_group_probability": p1,
                "one_layout_per_batch": True,
                "independent_times_per_example": True,
                "time_sampler": "uniform_canonical_original" if is_lsd else "native",
                "noise_sampler": "gaussian_original" if is_lsd else "native",
                "loss_reduction": "original_valid_token_mean" if is_lsd else "native",
                "ot": True if is_lsd else "inherit_native",
                "cfg": False,
                "meanflow": False,
                "persistent_cache": False,
                "lsd": {
                    "enabled": is_lsd,
                    "variant": "original_absolute_fd_v1" if is_lsd else None,
                    "recipe_source": self.original_recipe if is_lsd else None,
                    "p_fm": p_fm if is_lsd else None,
                    "fm_curriculum_steps": warmup if is_lsd else 0,
                    "total_train_steps": None,
                    "w_fm": 1.0,
                    "w_lsd": 0.5 if is_lsd else 0.0,
                    "finite_difference_delta": 0.01 if is_lsd else None,
                },
            },
            "inference": {
                "lifecycle_refinements": r,
                "cold_nfe": n,
                "warm_nfe": 1,
                "iterative_cold_start": not is_lsd or n == r,
                "consumption": "exact_C_confirmed",
                "output": "clean_front_only",
                "cache": "latent_per_session",
                "combined_rtc_async": "requires_separate_validation",
            },
            "integration_status": "configuration_only_not_adapted",
            "required_audits": [
                "native_time_sign_and_sampler", "native_prior_and_tail_independence",
                "actual_factory_and_action_head", "token_time_through_all_layers",
                "native_masks_and_loss_reduction", "checkpoint_key_shape_and_function_equivalence",
                "data_horizon_and_action_coordinates", "fresh_observation_prefix_and_suffix_kv_reset",
                "confirmed_consumption_and_session_reset", "fresh_versus_rolling_quality",
            ] + (["coupled_time_path_and_fd_precision", "teacher_stopgrad_and_packed_order"] if is_lsd else []),
            "warnings": warnings,
        }


def validate_resolved(config: dict[str, Any]) -> None:
    """按创建规则重算派生字段；编辑 JSON 不能把冲突带进执行器。"""
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("未知配置 schema")
    try:
        g, tr, inf, model = config["geometry"], config["training"], config["inference"], config["model"]
        expected = Selection(
            preset=config["preset"], policy_name=config["baseline_policy"],
            horizon=g["horizon"], execution_chunks=g["execution_chunks"], actions_per_output=g["actions_per_output"],
            initialization=config["initialization"], checkpoint_horizon=config["checkpoint_horizon"],
            allow_horizon_change=config["allow_horizon_change"], group_domain=tr["group_domain"],
            single_group_probability=tr["single_group_probability"],
            native_time_convention=model["native_time_convention"],
            original_recipe=tr["lsd"]["recipe_source"] or "core",
            refinements=inf["lifecycle_refinements"], cold_steps=inf["cold_nfe"],
        ).resolve()
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"配置缺失或类型错误：{error}") from error
    # 这些字段不能手工绕过；状态/审计证据由宿主记录在另一个文件中。
    for key in ("registered_policy", "model", "training", "inference"):
        if json.dumps(config.get(key), sort_keys=True, allow_nan=False) != json.dumps(expected[key], sort_keys=True, allow_nan=False):
            raise ValueError(f"配置字段 {key} 与所选方案不一致，请通过创建器重生成")
    if config.get("integration_status") != "configuration_only_not_adapted":
        raise ValueError("不要在配置中伪造已验证状态；请提供独立宿主验证报告")
