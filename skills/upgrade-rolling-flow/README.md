# RollFlow Skill v1.0 — 可选配置、瞬时版优先

这是完整的可移植 **Agent Skill＋参考代码＋测试包**，不是所有 VLA 的自动代码转换器。解压后保留 `upgrade-rolling-flow/` 目录名，把该目录交给代码 agent 或放到其支持的项目级 Skills 位置；本包未自动安装到任何系统目录。

## 最快尝鲜

```bash
cd upgrade-rolling-flow
python scripts/configure.py --interactive
```

按回车采用默认：**瞬时流匹配、预测窗口 H64、执行分块 M4、每次输出并执行 C16、无 LSD、无 CFG**。

也可不交互：

```bash
python scripts/configure.py --yes --policy-name dm05 --output dm05_rollflow.json
```

| 参数 | 默认 | 说明 |
|---|---:|---|
| 预测窗口 H | 64 | 新初始化的默认；旧 checkpoint 优先保留原长度 |
| 执行块数 M | 4 | `M = H/C` |
| 每轮输出 C | 16 | 返回已完成去噪的前段；后端 latent 不是可直接执行的动作 |
| 每动作累计细化 R | 4 | 分摊在多个周期；不是每周期执行四次前向 |
| 冷启动 NFE | 4 | 建立阶梯缓存 |
| 稳态每轮 NFE | 1 | 消费完上一段 C 后移位、补噪声、一次联合前向 |
| 训练时间分组 G | 1/2/4/8/16/32/64 | 每个 batch 任选一种；不是固定成四组 |
| 训练目标 | 原瞬时 FM | 原 loss、noise/time sampler 和动作处理需保留 |

创建器只写新配置；拒绝覆盖同名配置，不改目标模型。`from_native` 必须在宿主审计中解析成真实时间约定。配置检查通过不等于模型已适配。

### 已训练权重与新训练

已核实原策略 H12，可直接保留 H12/C3：

```bash
python scripts/configure.py --yes --policy-name my_policy \
  --initialization finetune --checkpoint-horizon 12 --output finetune_h12.json
```

H50 不适合默认 M4，显式选 M5/C10：

```bash
python scripts/configure.py --yes --policy-name my_policy \
  --initialization finetune --checkpoint-horizon 50 \
  --execution-chunks 5 --output finetune_h50.json
```

无策略 checkpoint：选 `--initialization scratch`，保留项目自己的骨干预训练约定。已有权重能加载，不等于没做 token-time 微调就有良好的 rolling 质量；“快速尝鲜”首先保证改动少、容易复现。

### 可选原始 LSD

```bash
python scripts/configure.py --yes --preset lsd-original \
  --original-recipe core --refinements 2 --cold-steps 2 \
  --policy-name my_policy --output lsd_r2.json
```

这明确选择原始 absolute-FD LSD。默认配方忠实保留原始 core 的 `p_fm=.7`、warmup=20000；`starvla-wrapper` 为 `.3` 和 5000。实际训练总步数必须在宿主配置中核实，不足 warmup 就不会进入 LSD。更多开销和更少采样步数是需要衡量的取舍，不承诺一步/两步等质。

## 在本机运行验证

```bash
python -m pip install -r requirements-test.txt
python scripts/run_checks.py
python examples/quickstart.py --steps 24 --output my_quickstart.json
```

尝鲜脚本创建真实的小型联合注意力 action head，进行严格权重加载、标量/逐 token 输出和梯度对照、合成数据更新、三次 cold/warm 输出。它**不下载、不冒充** π0、DM05、GR00T 或发布的 VLA checkpoint。正式质量和 GPU/机器人运行需要宿主环境。

## 文件导航

- `SKILL.md`：agent 的创建选择、迁移、验证和交付流程。
- `references/ORIGINAL_ROLLFLOW_ALGORITHM.md`：用户原始伪代码，原样保留。
- `references/original_sources/`：三份原始代码和 SHA256 清单。
- `references/ALGORITHM_AND_LIMITS.md`：时间约定、原始/实验 LSD 区别、已知局限。
- `references/DATAFLOW_AND_CACHE.md`：数据、前缀 KV、latent 缓存与消费协议。
- `references/MIGRATION_CONTRACT.md`：独立注册、权重、mask、基线对比准则。
- `references/VALIDATION_PROTOCOL.md`：验证等级、失败归因和再验证。
- `assets/`：瞬时 rolling、token-time、可选实验 LSD 参考内核及配置解析。
- `scripts/configure.py`：交互/非交互创建，纯标准库即可执行。
- `tools/scan_policy.py`：只读候选定位；不做自动改写或兼容性背书。
- `templates/HOST_VALIDATION.json`：每个实际策略的验证清单，初始全部未验证。
- `tests/`、`TEST_REPORT.md`、`reports/`：本次真实测试、日志和范围说明。

历史 v0.5 的报告单独保留。v1 的通过数只采用本次命令实际结果；没有继承过去对 XPolicy 的生产模型通过宣称。
