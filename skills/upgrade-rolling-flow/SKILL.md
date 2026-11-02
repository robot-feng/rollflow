---
name: upgrade-rolling-flow
description: Upgrade an existing robot or VLA flow-matching action policy to genuine per-action noise time and persistent rolling inference. Offer a quick instantaneous-FM preset or explicitly opt-in original RollFlow central-difference LSD, preserve baseline/checkpoints/pseudocode, create independently named adapters, and validate training, dataflow, time conditioning, cache semantics and generated code. Not a universal automatic source converter.
compatibility: Agent instructions are framework-neutral. Bundled executable references use Python 3.10+, PyTorch and pytest on CPU. Native VLA, JAX, GPU, RTC, distributed and robot integrations require their own environments and tests.
metadata:
  version: "1.0.0"
  default-preset: "quick-instant-h64-m4-c16"
---

# RollFlow 迁移 Skill v1.0

## 0. 创建时先让用户选，不把所有参数硬编码

用户已经指定的值直接沿用，不重复提问。未指定时，一次给出下面三个入口；用户说“默认”“最快尝鲜”就直接选第一项。

| 入口 | 默认配置 | 改动与成本 |
|---|---|---|
| **快速尝鲜（默认）** | 瞬时 FM；H=64；4 个执行块；每轮输出并执行 C=16；冷启动 4 次专家前向、稳态 1 次 | 保留瞬时速度目标；优先只改逐位置时间形状和执行器，不新增模型参数 |
| **原始 LSD（主动选择）** | 保留原始 FM＋绝对中心差分 LSD；配方来源明确选择 core 或 StarVLA wrapper | 需要端点/区间条件，更多训练计算；减少到 1/2 次细化的质量必须实测 |
| **自定义** | 可改 H、执行块数或 C、训练分组粒度、初始化方式 | 派生值自动计算，冲突配置直接拒绝 |

交互创建：`python scripts/configure.py --interactive`

直接采用默认：`python scripts/configure.py --yes --policy-name dm05 --output my_config.json`

创建器**只生成迁移配置，不改源码、不加载权重、不代表已完成适配**。配置保留未确定的原生时间方向供审计，不能猜它。

### 把四种“数量”分开

- `H`：模型预测窗口；`M=H/C`：执行块数；`C`：一次实际输出/消费的动作数。
- `G`：训练噪声时间分组数；quick 默认为 `Div(H)`，H64 得到 1/2/4/8/16/32/64，每个 batch 随机选一种，不同样本分别采时间。
- `R`：一个动作生命周期内累计细化次数；quick 固定 `R=M`。
- `N`：冷启动串行专家调用数；quick 固定 `N=M`；稳态每轮仅一次。

默认“分块 4”**不是**训练只允许四种时间，也不是每轮串行前向四次。

已有策略 checkpoint：优先读取并保留其 horizon/位置参数。H12 可保留 H12/C3/M4；H50 与默认 M4 不整除时必须让用户选合法块数，例如 M5/C10，不能偷偷补到 64。新初始化默认 H64。主干是否加载预训练 VLM 遵循原工程；“从零训练策略”不自动表示随机初始化 VLM。

## 1. 不可改写的算法来源

首先阅读 [原始算法规范](references/ORIGINAL_ROLLFLOW_ALGORITHM.md)。其中伪代码、时间方向、`K∈Div(H/C)`、单样本共用 ratio、FM curriculum 原样保留。三个上传源码位于 `references/original_sources/`，使用 `python tools/check_originals.py` 校验。

原始 LSD 为：两次 no-grad 自教师调用，随后一次打包的学生调用 `[LSD-, LSD+, full-FM]`，以轨迹切线拟合终点瞬时速度，并保留全 batch 的 local FM。打包不代表计算量只有一个普通 batch。

**禁止**把原始绝对 `t±δ` 替换成 `t±h(t-s)`、MeanFlow identity、JVP、额外 CFG，或者加入训练缓存却仍叫“原版”。`assets/lsd_flow_torch.py` 是历史保留的**显式直线路径实验版本**，不由默认创建器选择，不能冒充原版。

原始代码存在待审计局限，原样归档不等于全部正确。见 [算法与原始实现边界](references/ALGORITHM_AND_LIMITS.md)。

## 2. 只围绕三个必要改动面迁移

### A. 时间采样

quick 调用宿主原采样器生成 `[B*G]` 标量，恢复 `[B,G]` 后排序、重复到 `[B,H]`。这里只展平**采样/时间编码**，不将动作序列拆成独立模型 batch。

保留时间取值范围、变换顺序、dtype 和 RNG；noise-at-zero 近端时间高、远端低，noise-at-one 相反。排序改变位置条件下的边缘分布，这是一项明确的训练改动。强制 G=1、固定相同 noise/time 时，输出、loss 和原参数梯度应与原生路径一致。

区分 quick 的 `Div(H)` 与原始 LSD 的 `Div(H/C)`。不得为了统一默认值改原版。分组单位可在配置中选择，但 original LSD 不能选择 action-level 扩展后仍声称复现原版。

### B. 模型支持真正逐位置时间

追踪 `[B] → [B,H] → embedding → action/state 融合 → 每层 AdaLN/AdaRMS → 最终 norm/projection`。使用原参数和原模块，能只改广播就不新增层。保证整个动作序列仍在原来的 attention 中联合计算。

禁止以 `B→B*G` 重复整条专家、每次用不同标量时间再拼片段冒充单次 token-time 前向；这不是一般联合场的等价实现，也不能按一次正常 batch 计算成本。

动作位置索引、噪声时间、LSD 源/目标端点、机器人实际控制时间四者分别记录。对 state/history/future/image/register tokens 给出显式时间归属；一个 token 池化多个不同噪声动作时，不得随意取平均时间。

逐位置编码的存在不证明整个网络可用。对 native time bucket、全局调制、条件拼接、每个最终输出层都做 shape/梯度测试。LSD 的有限差分与离散 bucket 会有冲突，必须单独检查，不能静默取消量化。

### C. 训练与 rolling 执行器

保留原瞬时速度定义、插值路径、normalizer、动作坐标、mask 和 loss reduction。用 `w_lsd=0` 的真正一前向路径实现 quick，不能额外建立教师图。使用原噪声先验；新 tail 独立采样只适用于已验证的时序独立先验。

缓存只保存动作 latent；不跨轮复用带旧位置的 suffix hidden/KV。每轮使用最新观测重新准备条件，当前轮内复用 prefix，suffix KV 不能越积越长。重置后必须冷启动。

严格使用已消费的 C 推进窗口。不能以读取一次输出、收到一次 RPC 或提前预测一次来代表执行完成。缓存按 stream/episode/model/normalizer/动作坐标系绑定；当前引用内核是同步、整块消费、batch 行共同推进，不是自动异步执行器。见 [数据流和缓存协议](references/DATAFLOW_AND_CACHE.md)。

## 3. 独立注册，原算法继续可用

新增 `<policy>_rollflow_instant`，显式选择 LSD 时新增 `<policy>_rollflow_lsd`，原注册项、源文件、checkpoint 路径保持不变。

优先子类/组合和小范围 shape helper；不复制整套框架，不 monkey-patch 全局注册器，不用宽泛 `strict=False` 隐藏载入错误。新增 LSD 参数要列出精确白名单；共享参数严格加载。导入顺序、同名冲突、保存重载、优化器/EMA 参数集合都需要测试。

原版、grouped-fresh、instant-rolling 使用清晰的独立配置，记录相同数据和训练预算。原有训练/processor/预处理与控制后处理不另起一套。

## 4. 实际使用 Skill 的验证闭环

1. **定位**：锁定仓库版本和依赖，沿 factory/registry 确认实际策略、具体子分支和 action head。`tools/scan_policy.py` 只提供候选位置，不输出“兼容/通过”。
2. **基准**：固定输入、noise/time、RNG 和 mode，记录原输出、loss、梯度、state-dict。使用 [宿主记录模板](templates/HOST_VALIDATION.json)。
3. **最小实现**：先实现独立 instant adapter，保持 G=1；再跑每一种 G；随后接 fresh/rolling。先通过正确性，再做 compile/mixed precision 加速。
4. **原生验证**：真实原生模块 forward/backward、实际生成补丁重放、严格加载/保存、optimizer step、context 梯度、完整数据到动作输出。stub/AST 只能记对应证据，不能记完整策略成功。
5. **专项**：故障不提交、reset、输出不别名、未消费拒绝、改变 batch/坐标/normalizer 拒绝；LSD 额外验证教师停梯度、packed 排序、差分精度、整个向量对角条件。
6. **质量**：已有策略进行 grouped 微调，新策略可从原初始化直接 grouped 训练。报告 cold＋连续 warm、突变、同预算 fresh、1/2/4/8 调用质量；不要根据 smoke loss 下降宣称控制有效。
7. **反馈**：每个失败记录 reproduction、根因归属（Skill/宿主实现/测试/算法假设）、最小修复及回归用例。在未改原版的新副本上重放；通过后才交付目标代码。

用 `python scripts/run_checks.py` 运行本包测试；用 `python examples/quickstart.py` 跑可执行的小型联合注意力尝鲜。**这些不是 XPolicy 各生产策略的验证报告。**

## 5. 交付和验收

每个目标交付原版保留说明、独立注册名、resolved 配置、精确 diff/源码 hash、训练/推理入口、实际命令与日志、证据等级和未测项。标准见 [迁移准则](references/MIGRATION_CONTRACT.md) 与 [验证规则](references/VALIDATION_PROTOCOL.md)。

主入口默认不加入 CFG、额外 teacher 网络、JVP、persistent training cache 或 adjacent-window sampler。LSD 是可选升级，不是迁移 instant 的前置条件。原版资料和实验性扩展必须分开。

通过配置检查不等于通过模型测试；通过小型原生头不等于通过生产 checkpoint；通过离线动作误差不等于通过仿真或机器人。不能承诺“适配所有框架而无条件正确”。
