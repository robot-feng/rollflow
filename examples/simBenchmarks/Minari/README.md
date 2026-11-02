# Minari：纯状态 FM / Rolling / LSD 对照

当前快速多环境筛查结果见 [Reacher H32/C4 三组报告](results/REACHER_WIDTH64_20261031.md)。Reacher 上 LSD 瞬时速度与纯 rolling FM 接近；平均速度在 K=2/4/8 明显较差。该实验每组只有一个训练种子，结论仍需在 Hopper 等更长任务和多个训练种子上复核。

## 目的与结构

用小数据、实际动作头快速诊断收敛，再用真实任务的 checkpoint/输入复核原因。
Minari 负责数据和环境恢复，模型学习的是行为克隆，不是额外的离线强化学习目标。

- `starVLA/model/framework/StateAction/StateFlow.py`：薄框架，状态直接作为 cross-attention context，同时进入原有 state encoder；没有 VLM，也没有新增可学习投影层。
- `train_files/`：单组训练、三组对照、三组续训、LSD 参数扫描，以及可直接运行的 Pusher H32/C4 配置。
- `eval_files/`：checkpoint 评估、训练曲线汇总与少步配对比较。
- `dataset_tools/`：本地 Pendulum 示范数据的采集入口。
- 顶层 `run.py`：数据划分、训练、回放与 Gymnasium 环境闭环的共同实现；`run_suite.py`、`run_continuation_suite.py`、`run_lsd_delta_suite.py` 和 `summarize.py` / `compare_few_step.py` 为现有实现。

新入口与 LIBERO 一样按用途分目录。顶层实现和原命令保留：正在运行的进程、排队脚本、测试和既有实验源码快照都依赖这些路径。新入口只是调用同一实现，不引入第二套算法。

```text
Minari/
├── README.md / TASK_SELECTION.md / requirements-pusher.txt
├── train_files/         # 训练、续训、三组和 LSD 扫描入口
├── eval_files/          # 离线回放、仿真评估和对比报告入口
├── dataset_tools/       # 采集入口
├── run.py              # 共同数据、训练和评估实现
└── run_* / summarize.py / compare_few_step.py  # 既有实现及兼容路径
```

一次三组实验的输出放在用户指定的独立目录中。`suite.json` 记录各组命令和状态；每组目录包含 `config.json`、`manifest.json`、`metrics.jsonl`、`last.pt`、`source/` 和 `evaluation.json`。评估生成的 `replay_*.npz` 保存动作预测和数据集动作，比较脚本另建目录写入报告和图片。

普通 VLA trainer 的数据入口尚未支持 Minari，使用这里的训练入口；它直接调用框架和实际动作头，不复制训练损失、采样器或 rolling 积分器。

|variant|动作头|训练|推理|
|---|---|---|---|
|`fm`|`GR00T_ActionHeader.FlowmatchingActionHead`|原版 FM，Beta 时间、整数桶|每次重新生成 H，执行前 C|
|`rolling_fm`|`GR00T_RollFlow_ActionHeader.RollFlowActionHead`|原生阶梯 FM，`w_lsd=0`|瞬时速度，保留 rolling cache|
|`rolling_lsd`|同一个 RollFlowActionHead|同样阶梯采样 + LSD|平均速度，同时评估瞬时速度|

第二组没有使用 `GR00T_RollFlowMatching_ActionHeader.py`。

## 对照约束

1. 按完整 episode 划分 train/validation/test，状态统计仅使用 train；动作按环境 Box 边界线性归一化。
2. 三组均采样完整 H 窗口，不跨 episode。原版头没有 padding loss mask，因此共同排除尾部不完整训练窗口。回放评估仍覆盖完整测试 episode。
3. 所有共享参数从同一原版 FM 初始化加载，保存 SHA256；RollFlow 的额外 interval 编码沿用现有兼容加载逻辑。保留零区间编码，除非显式指定 `--skip-zero-interval`。
4. `--batch 8 --repeat 4` 是 8 个原始观测、32 个 diffusion 样本。复制顺序与 QwenGR00T 的 `.repeat(4,1,1)` 一致，复制后由动作头重新采样噪声、时间。RollFlow 的 K 仍是每 batch 共用，不能声称 32 个 K 独立。
5. 默认 H32/C8、`p_k1=0`，rolling 随机选择 K=2 或 4 的有序阶梯；不是固定 K=4。部署默认 K=1/4 分别评估，可用 `--eval-nfe 1 2 4` 添加中间等级。原版 FM 保留原版时间分布，不人为改成阶梯。
6. Rolling 两组每步使用相同数据、噪声、时间种子和初始化。FM warmup 内 loss/梯度已检查一致。每步重新设种子，评估与额外 LSD forward 不影响下一步随机数。
7. `--precision bf16` 为 BF16 autocast + FP32 主参数及 AdamW 状态。它隔离前向精度影响，不等价复现 DeepSpeed 的所有行为。
8. 小尺寸通过进程内临时 `diagnostic` preset 构造，随后恢复 preset 字典。原版头的 encoder 维度依赖命名 preset，仅覆盖 diffusion 配置会维度不一致；生产模型源码未为实验改动。

## 环境与数据

Pusher 的已验证 Minari / Gymnasium / MuJoCo 版本列在 [requirements-pusher.txt](requirements-pusher.txt)。建议在独立环境中安装，并先按仓库要求安装 PyTorch 和 starVLA 依赖：

```bash
python -m pip install -r examples/simBenchmarks/Minari/requirements-pusher.txt
```

这个文件仅覆盖 Pusher 实验依赖，不安装 GPU 版 PyTorch；Pendulum 本地 MPC 数据采集还需要 CPU 版 JAX/JAXLIB。已有训练环境不要直接升级依赖。

Pusher expert-v0 的环境元数据指向 `Pusher-v5`，要求 Gymnasium ≥1.0 和 MuJoCo 3.2.3。运行前要在**实际启动脚本的解释器**里确认导入版本；只看 `pip show` 不够，因为带 `include-system-site-packages=true` 的 venv 可能优先导入基础环境里的 Gymnasium 0.29.1。可先执行下面的预检，确认 `recover_environment()`、reset 和 step 都通过，再开始长评估：

```bash
MINARI_DATASETS_PATH=/path/to/minari_datasets python - <<'PY'
import gymnasium, minari, mujoco
print("Gymnasium", gymnasium.__version__, gymnasium.__file__)
print("MuJoCo", mujoco.__version__, mujoco.__file__)
ds = minari.load_dataset("mujoco/pusher/expert-v0")
env = ds.recover_environment()
obs, _ = env.reset(seed=70000)
obs, reward, terminated, truncated, _ = env.step(env.action_space.sample() * 0)
print(env.spec.id, obs.shape, reward, terminated or truncated)
env.close()
PY
```

如果版本或环境入口不匹配，先修复隔离环境或 `PYTHONPATH`，不要把环境恢复失败记为策略评估失败，也不要用另一 Gymnasium/MuJoCo 版本的 return 代替正式对照。

先激活安装了 starVLA 和 Minari 的 Python 环境；下例直接使用 `python`。
新增依赖为 Minari 0.5.3、CPU JAX/JAXLIB 0.4.35，未更改训练环境。

数据根：`/path/to/minari_datasets`，也可通过 `MINARI_DATASET_ROOT` 指定。
数据 ID：`rollflow/pendulum-mpc-v0`，16 条、3200 transitions，来自真实 Pendulum-v1 的 random-shooting MPC。
采集平均 return 约 -168.33，仅作本地诊断示范，不是官方 expert 或最优策略。
数据接口说明：[Minari basic usage](https://minari.farama.org/content/basic_usage/)。

如需独立采集，使用新的 dataset ID，避免覆盖已有数据：

```bash
JAX_PLATFORMS=cpu python \
  examples/simBenchmarks/Minari/run.py collect \
  --dataset rollflow/pendulum-mpc-local-v0 --episodes 24
```

支持 flat Box state 和 continuous vector Box action 的 Minari 数据；若有 Dict observation，先显式编写字段适配，不隐式改变状态语义。

## Pusher 对照快速入口

下例从仓库根目录启动；将 Python 和数据目录换成自己的路径。数据 ID 已存在于 `MINARI_DATASET_ROOT` 时无需下载。默认在 CPU 上运行，每进程两线程。

```bash
export STARVLA_PYTHON=/path/to/minari-env/bin/python
export MINARI_DATASET_ROOT=/path/to/minari-datasets
export OUTPUT_DIR=/path/to/new-pusher-suite
CUDA_VISIBLE_DEVICES='' bash examples/simBenchmarks/Minari/train_files/run_pusher_h32_c4.sh
```

默认 Pusher expert 前 200 条 episode、H32/C4、batch 8、repeat 4、45K、seed42，评估 K=1/2/4/8。训练原版 FM 和一个 RollFlow 模型：后者前 30K 保持 source time 等于 terminal time，使用瞬时速度评估；后 15K 逐渐启用蒸馏，再使用均值及瞬时速度评估。需要额外的独立纯 rolling 对照时，追加 `--variants fm rolling_fm rolling_lsd`。结果依次写入 `OUTPUT_DIR`，已存在的 suite 不会覆盖。训练时长改变后，默认 FM warmup 随之取总步数的 2/3。`TRAIN_STEPS`、`FM_WARMUP_STEPS`、`LSD_RAMP_STEPS`、`TRAIN_SEED`、`BATCH_SIZE`、`REPEAT`、`THREADS` 可通过环境变量调整，其他参数直接追加到命令末尾。不同参数会产生不同实验，比较时应统一配置。45K 是初始预算，仍需通过中间 checkpoint 和续训判断是否达到平台。

其他规范入口，参数与对应旧脚本一致：

```bash
bash examples/simBenchmarks/Minari/train_files/train_minari.sh --help
bash examples/simBenchmarks/Minari/train_files/run_three_way.sh --help
bash examples/simBenchmarks/Minari/train_files/resume_three_way.sh --help
bash examples/simBenchmarks/Minari/train_files/run_lsd_sweep.sh --help
bash examples/simBenchmarks/Minari/eval_files/eval_minari.sh --help
bash examples/simBenchmarks/Minari/eval_files/eval_checkpoints.sh --help
bash examples/simBenchmarks/Minari/eval_files/compare_few_step.sh --help
bash examples/simBenchmarks/Minari/dataset_tools/collect_pendulum.sh --help
```

checkpoint 评估入口示例：

```bash
CUDA_VISIBLE_DEVICES='' bash examples/simBenchmarks/Minari/eval_files/eval_minari.sh \
  --dataset-root "$MINARI_DATASET_ROOT" --checkpoint "$OUTPUT_DIR/rolling_lsd_fp32_seed42/last.pt" \
  --output /path/to/new-evaluation --eval-episodes 20 --eval-max-steps 100 --eval-nfe 1 2 4 8
```

评估从 checkpoint 恢复训练配置和数据划分；评估目录应使用新路径。它会同时输出回放误差、真实环境闭环回报和实际 DiT 调用次数。

检查蒸馏是否趋于稳定时，可以对同一训练过程的多个 checkpoint 使用相同评估设置：

```bash
RUN_DIR=/path/to/rolling_lsd_run OUTPUT_DIR=/path/to/new-progress-eval \
CHECKPOINT_STEPS='30000 35000 40000 45000' \
CUDA_VISIBLE_DEVICES='' bash examples/simBenchmarks/Minari/eval_files/eval_checkpoints.sh \
  --dataset-root "$MINARI_DATASET_ROOT" --eval-episodes 10 --eval-max-steps 100 --eval-nfe 1 2 4 8

"$STARVLA_PYTHON" examples/simBenchmarks/Minari/eval_files/summarize_checkpoints.py \
  --input /path/to/new-progress-eval --output /path/to/new-progress-report
```

这些检查点应来自同一次训练。仅 FM 验证 loss 稳定不能证明平均速度策略稳定，仍需比较各 checkpoint 的均值速度回报和回放误差。

可对训练日志做稳定性快照（续训的日志用逗号连接）：

```bash
"$STARVLA_PYTHON" examples/simBenchmarks/Minari/eval_files/summarize_stability.py \
  --run "experiment=/path/to/train45/metrics.jsonl,/path/to/train60/metrics.jsonl" \
  --after 30000 --interval 5000 --clip 1 --output /path/to/new-stability-report
```

报告统计已记录的裁剪前梯度范数、超阈值记录点和固定 FM 验证误差。日志通常每 250 步记录一次，超阈值记录点数不是全部训练步的超阈值次数，也不能用 FM 验证误差替代滚动策略评估。

同一 RollFlow 模型的训练前后可直接比较：30K checkpoint 选 `instant_k1/2/4/8`，蒸馏后 45K、60K 等 checkpoint 选 `average_k1/2/4/8`。续训目录可与原实验目录一起传给汇总脚本（例如 `--input /path/to/progress-45k /path/to/progress-60k`）；脚本核对环境种子和测试 episode 一致、步数不重复。原版 FM 是独立参照，其 Beta 时间采样、全局条件化与 RollFlow 的阶梯时间和逐 token 条件化仍有差异，不能视为数学上完全相同。

评估默认从环境 seed 60000 开始；使用 `--eval-seed-start 70000 --eval-episodes 50` 可运行不同的环境种子组。比较的各模型必须使用相同种子组；重复查看和选择模型后，应用尚未用于选择的种子单独复评。环境种子不同不等于训练种子独立，仍须多次独立训练。

如果要比较多个 LSD 权重，可以从同一个纯 FM warmup 边界 checkpoint 分支，避免重复训练前 30K。分支只修改 LSD 权重配置，保留模型、优化器、调度器和数据划分，并保存源文件与分支的 SHA256：

```bash
"$STARVLA_PYTHON" examples/simBenchmarks/Minari/train_files/branch_lsd.py \
  --checkpoint /path/to/rolling_lsd_run/step_30000.pt --w-lsd 0.2 --output /path/to/new-branch
CUDA_VISIBLE_DEVICES='' bash examples/simBenchmarks/Minari/train_files/train_minari.sh \
  --resume-from /path/to/new-branch/branch.pt --additional-steps 15000 --output /path/to/new-branch-training
```

分支只允许从保存的 FM warmup 边界创建；续训仍使用原训练计划中的日程和学习率。`branch_manifest.json` 记录配置变化，续训继续核对原始源码和数据。不同权重的比较仍需多个训练种子与相同的闭环评估。

可以在创建分支时传入 `--precision bf16`（或 `fp32`）来比较蒸馏阶段的 autocast 精度；checkpoint 保留原 FM 阶段模型和优化器状态，分支 manifest 记录修改。这里 BF16 指 autocast 和 FP32 模型/优化器参数，并不等同 DeepSpeed 的 BF16 参数训练。比较训练精度时，评估入口可使用 `--eval-precision fp32` 固定推理精度。
如需研究单时间组与多时间组的混合比例，可显式添加 `--p-k1 0.5`；这是训练采样变化，会在分支记录中标明。训练时间组数 K 与部署时 refinement K 是不同的量。

同一 K 下配对比较蒸馏前后的 return 和回放 MSE：

```bash
"$STARVLA_PYTHON" examples/simBenchmarks/Minari/eval_files/compare_training_phases.py \
  --reference /path/to/progress-45k/step_30000/evaluation.json \
  --candidates /path/to/progress-45k/step_45000/evaluation.json /path/to/progress-60k/step_60000/evaluation.json \
  --output /path/to/new-phase-comparison
```

脚本检查相同的环境种子和测试 episode，但使用者仍应核对原始 checkpoint 与续训记录来自同一训练链；它不会根据文件名推断训练血缘。

## 训练与评估

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 python examples/simBenchmarks/Minari/run_suite.py \
 --output /path/to/my_minari_suite --steps 10000 \
 --fm-warmup 2000 --lsd-ramp 1000 --batch 8 --repeat 4 \
 --width 32 --layers 2 --seeds 42 --precisions fp32 bf16
```

默认 `w_lsd=.05`、velocity loss 单位、delta=.01；这是探测起点，未证明是最优值。OT 默认关闭以统一三组基线，rolling 可显式开启；不把 OT 当作已知故障原因。

每 1000 步保存完整模型/优化器/scheduler 状态。非有限 loss/梯度在更新前保存现场并失败退出。
需要观察蒸馏收敛时，可以从 checkpoint 续训到新目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 python examples/simBenchmarks/Minari/run.py train \
 --resume-from /path/to/run/last.pt --additional-steps 10000 \
 --output /path/to/new_continuation
```

续训保持原模型配置、数据划分、归一化、batch、repeat 和训练种子，恢复 AdamW 与 scheduler。超过原计划步数后，学习率保持原计划的最终值，p_fm 保持原计划的最终值，不重新开始衰减。新的 checkpoint 保存数据内容哈希；较早 checkpoint 没有该哈希时，只能核对 episode 划分和归一化统计，日志会明确记录。若动作头或框架源码与原实验不同，默认拒绝续训；显式使用 `--allow-resume-code-change` 时记录变更，此时它是新的干预实验。

### 实验性干预的续训

部署形时钟和端点一致性是仅在当前训练进程启用的实验，不会写进模型权重。首次从普通 checkpoint 开始实验，以及后续从该实验 checkpoint 续训，都须使用相应入口：

```bash
python examples/simBenchmarks/Minari/train_files/train_deployment_clock_ablation.py train \
  --resume-from /path/to/clock-run/last.pt --additional-steps 15000 --output /path/to/clock-continued

python examples/simBenchmarks/Minari/train_files/train_endpoint_consistency_ablation.py train \
  --resume-from /path/to/endpoint-run/last.pt --additional-steps 15000 --output /path/to/endpoint-continued
```

每个运行目录保存实验入口源码及 JSON 摘要。续训会检查 checkpoint 记录的原输出目录中的干预标记，拒绝使用普通入口或另一干预入口，防止静默改变训练目标；`--allow-resume-code-change` 不会启用这些干预。请保留原输出目录和标记文件。检查不比较入口源码哈希，跨版本续训仍须人工核对各运行目录保存的源码和摘要。评估时无需启用训练干预。

端点一致性入口可在从普通 FM checkpoint 创建实验时附加 `--endpoint-weight 4` 等非负数，默认 1；这是额外的位移一致性损失系数，不是 LSD 权重。设为 0 会跳过整个端点分支，便于隔离 FM＋LSD。实验续训默认沿用原目录 JSON 中的权重，若显式传入不同权重会报错；要改变权重，应从相同的 FM checkpoint 创建新分支。

可选 `--endpoint-teacher frozen` 将首次分支使用的 checkpoint 动作头作为冻结的端点教师；默认 `self` 使用当前训练中的动作头。冻结教师只负责生成端点目标，并额外占用一份动作头的内存。续训继承教师类型，并检查教师 checkpoint 的哈希；因此冻结教师 checkpoint 必须保留。它属于实验干预，不改变生产 RollFlow 模型或 LSD 公式。

可选 `--endpoint-front-weight 4` 等正数增加端点损失中第一个可执行 chunk 的相对权重，默认 1 表示所有有效 token 等权。归一化使用有效权重之和；例如完整无填充的 K=2 阶梯，首 chunk 的损失权重从 1/2 改为 4/5，而 K=1 时只有首 chunk 有效，加权不会改变损失。续训沿用原权重并拒绝静默更改。这是训练实验，不改变推理 chunk 或生产模型。

可选 `--lsd-normalization valid` 将 LSD 对有效 token 的条件平均乘以活跃 token 占有效 token 的比例，也就是按所有有效 token 平均；默认 `active` 保留现有条件平均。前者在活跃 token 很少时降低该批次的有效 LSD 系数，因此同一个 `w_lsd` 在两种模式下强度不同。日志中的 `lsd_weight` 和 `weighted_lsd_loss` 记录实际有效系数和加权损失。此选项仅用于实验入口，续训沿用并禁止静默切换。

`--lsd-normalization ramp10` 使用条件平均乘以 `min(1, 活跃比例 / 0.1)`：当活跃比例低于 10% 时平滑减小有效权重，达到 10% 后与 `active` 一致。它是待验证的训练干预，并非生产默认。

`--lsd-robust-threshold 1` 将 LSD 的速度残差平方替换为伪 Huber 损失 `2c²(sqrt(1 + (e/c)²) - 1)`，其中 `c` 以速度单位计，小残差时接近平方，大残差的梯度受限。FM 损失、时间采样和推理不因该选项改变；它改变 LSD 目标，不是原算法的严格等价实现。默认阈值为 0，表示原平方损失。续训继承阈值并拒绝静默切换；记录中的 `lsd_loss` 为所选损失，不同阈值的数值不能直接当成同一 MSE 比较。该选项仍处于实验验证阶段。

`--endpoint-source cache` 是端点一致性项的实验选项：对采样的当前观测，按真实轨迹提取之前 K 个 chunk 间隔的观测，用当前学生的平均速度滚动 K 次，从缓存移位得到端点监督的输入；缓存构造不反向传播。历史不足 K 个 chunk 的样本不计入端点损失。默认 `path` 使用动作标签和高斯噪声的线性插值，原 FM 和 LSD 两项仍使用各自原来的输入；此选项并非持久训练缓存，也不改变推理。续训保留该配置，评估必须按真实滚动流程检验。轨迹记录观测不等于闭环模型自身生成的观测，不能把此实验视为闭环数据训练。

搭配 `--endpoint-source cache --execution-loss-weight 0.1` 可额外监督首个可执行 chunk 的预测动作与数据集动作之间的 MSE；默认 0 关闭。该项与 FM、LSD、端点一致性一同使用，权重随蒸馏阶段前 1000 步上升，历史不足和 padding 不参与；续训继承并拒绝切换。它是额外的动作监督，不是原始 LSD 损失，目前只用于研究，不能据此推断真机效果。

可选 `--execution-k-probs 0.5,0.3,0.15,0.05` 设置该首 chunk 监督采样的推理细化步数 K=1/2/4/8 概率；默认均匀分布以保持旧实验行为。horizon 不支持的 K 会被排除后对有效 K 重归一化；若有效 K 全部为零概率则报错。该选项只改变辅助执行损失所用的 K，不改变基础 FM/LSD 的阶梯时间采样或部署推理；概率写入实验摘要并在续训时锁定。它用于检验针对低步数的训练权重能否缩小 1/2 步与 4/8 步差距，尚无性能结论。

`--clock-sampling original` 保留基础 checkpoint 的训练时钟/K 采样，仅端点或首 chunk 辅助目标使用部署形阶梯；适合隔离辅助损失效果。`mixed` 将部分非对角训练样本换成部署形时钟，并均匀重采 K=1/2/4/8；`edge` 进一步把被替换样本放到上端边界附近，保持 FM 行和随机数消耗。后两者同时改变时钟支持/K 分布，不能用于单独归因辅助损失。默认 `mixed` 保持旧实验行为，续训继承并拒绝更改。上述采样仅用于实验，尚未证明它能改善闭环效果。

LSD 最终性能使用 final checkpoint 评估；`best_fm.pt` 仅按固定 FM 指标选取，不代表最优平均速度模型。

```bash
CUDA_VISIBLE_DEVICES='' python \
 examples/simBenchmarks/Minari/run.py eval \
 --checkpoint /path/to/run/last.pt --output /path/to/new_eval \
 --eval-episodes 20 --eval-nfe 1 2 4

python \
 examples/simBenchmarks/Minari/summarize.py /path/to/suite
```

比较少步和多步策略时，用环境种子与回放 episode 配对，单独看平均回报、回放误差和实际调用数：

```bash
python \
 examples/simBenchmarks/Minari/compare_few_step.py \
 --reference /path/to/fm/evaluation.json --reference-mode instant_k4 \
 --candidates /path/to/rolling_fm/evaluation.json /path/to/rolling_lsd/evaluation.json \
 --output /path/to/comparison
```

旧结果若没有显式保存环境种子，确认其源码使用 60000 起的连续种子后，才添加 `--legacy-runner-seeds`。报告中的配对区间只描述环境种子变化；需多个独立训练种子判断训练稳定性。

推理每次执行整个 C，再重新观测；rolling 只在 episode 边界 reset。离线回放使用记录状态，每 C 帧更新条件；闭环环境使用动作执行后的真实状态。保存每个关节预测/标签 NPZ、return、裁剪比例、调用次数与耗时。

`eval --eval-precision fp32` 或 `bf16` 可以只改变已有权重的推理精度；省略时使用保存的训练精度。评估目录记录 checkpoint SHA256、训练/推理精度和环境种子，便于区分训练退化与推理数值问题。

## 如何判定问题

- 同时看固定验证 FM loss、打乱状态后的 loss、零速度基线、预测 RMS、裁剪前梯度范数。不能只看随机训练 loss 或 FM+LSD 总和。
- 新实验每次记录指标时，还扫描固定 33 个对角时刻，记录时间编码 RMS、AdaLN 和输出层调制幅度（`clock_*`）。这个扫描不抽取随机数，不改变训练下一步采样。
- 若只有 LSD 组恶化，在相同 checkpoint 上拆分 FM/LSD 梯度及平均/瞬时推理；若 FM rolling 也恶化，优先比较模型、精度、优化器与输入条件。
- 若小模型均正常，继续真实尺寸及历史 checkpoint 固定输入诊断；小任务通过不排除真实任务问题。
- 三组达到平台后，再用多个训练种子和更多独立环境种子比较。没有平台时继续训练，不根据同一短预算断言蒸馏无效。
- K 表示 refinement 等级，不是 rolling 每个 chunk 的网络调用数。冷启动和热态分开解读，比较少步效果必须同时看实测调用数、耗时和环境 return。

## 已完成的首轮诊断

- 原 tmux：`rollflow_minari_three_arm_20261027`（队列已完成，tmux 会话可能随进程退出）
- 输出：写入运行命令指定的输出目录；本地 checkpoint 和数据未纳入仓库。
- 三组 × FP32/BF16 × seed42，每组 10000 步；CPU 两线程顺序执行。
- 先完成三组 × 两种精度的 12 步全流程检查，另有单元测试。检查通过仅证明实验链路可运行，不能说明长程稳定或控制性能良好。
- 六组 10K 均已完成；当前 LSD 均值速度没有达到瞬时速度表现，BF16 LSD 尤其弱。详细数字与限制见对应运行目录 `report.md`，不能以损失未发散就宣称蒸馏成功。
- 历史权重和真实尺寸干预记录：`docs/diagnostics/rollflow_collapse_intervention_20261027.md`。
