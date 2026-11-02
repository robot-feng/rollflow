# 算法、版本与局限

## 四个不同的承诺

- **原始算法保留**：上传 docstring 和三个源码文件不改一字，用 SHA256 验证。这是出处，不是正确性背书。
- **瞬时迁移兼容**：G=1 固定同 noise/time 下 native output/loss/梯度等价；没有改变参数结构。真实宿主必须实测。
- **新的 token-time 能力**：只改变每位置时间条件，仍联合建模整个动作窗口。可运行不自动表示泛化良好。
- **LSD 少步质量**：有独立训练与生成质量证据才能声明；不能由前三项推出。

## 方向和单位

noise-at-zero: `x=(1-t)*noise+t*action`, `v=action-noise`, Euler 增量为正。
noise-at-one: `x=t*noise+(1-t)*action`, `v=noise-action`, Euler 增量为负。

取反时间而不取反速度是错误迁移。连续积分时间与 bucket 输入分开，保留原倍数/取整规则；原 time sampler 的略微越界不能静默 clamp。非线性 sigma 路径、epsilon/v-pred/x0 目标、rectified-flow 权重等需要宿主公式，不能一律替换成这两个线性式。

## 原始 LSD 与直线路径实验不相同

原始实现固定源点，扰动激活 token 的目标时间 `t±δ*a`。它逼近的是向量时间场沿 active vector a 的方向导数。若动作 token 耦合，各位置 derivative 含交叉项，不能解释成只对本位置求偏导。

`assets/lsd_flow_torch.py` 则沿 `d=t-s` 使用 `t±h*d`，这是另一条路径。相同端点不能独立确定任意多时间流的传输路径，不能以一般 scalar MeanFlow/LSD 理论直接证明所有向量扩展。测试包含两种方向在耦合场上产生不同数值的反例。默认 quick 不受 interval-path 扩展的选择影响。

原始 active-token 截断会改变差分方向；被排除的非对角 token 仍可能参与 attention。源码注释称 inactive 为 diagonal，但“差分不扰动”不一定意味着 `s=t`。因此原版复现和普适耦合场的理论证明必须分开。不要为了让断言过关偷偷更换 stencil。

## s=t 的意义

整个向量端点相等时，区间位移为零，连续且正确训练的平均速度对角分支对应当前瞬时速度；不意味着永久等于训练前的旧权重，也不代表只令某个 token 端点相等就能隔绝其他 token 的区间条件。

原始 DiT 默认 `skip_zero_interval_embedding=False`，且缺失 interval 参数时会复制旧时间编码器。`embedding(0)` 通常不为零，因此严格的旧函数等价不能仅由权重可加载推出。`skip_zero_interval_embedding=True` 只是对角归零的分支选择，可能在零点有不连续；新 gap-gated 条件是另一个 schema，必须另名记录/比较，不能声称字节复现。

## 采样覆盖不是完整匹配

quick: `G∈Div(H)`，原 sampler，单布局/batch。original LSD: `K∈Div(H/C)`，优先 K1、Uniform target、共享 ratio、FM curriculum；core 与 wrapper 的 p_fm/warmup 默认值不同。

随机 ratio 的 `s=ρt` 不覆盖所有 warm 非径向端点。训练插值 latent 与多次 warm 的模型生成 latent 也有分布差距。有限步 Euler 即使用理想瞬时场仍有离散误差。需要 grouped-fresh/rolling 同权重对比、连续 warm、突变/reset 和真实数据验证；不能称“完全训练推理对齐”。

## 原始实现还不能直接用于任意生产会话

- `step()` 假定每次调用前已完整消费 C，未提供 session/request 身份或执行确认。
- 原始 `flow_map` 的 `0*NaN` 可能污染不动的 token；新参考内核单独用 where 保护，原始归档不改。
- 原始 select/pack helper 按首维等于 B 猜测 batch；静态矩阵恰好首维 B 或特殊 Cache 对象可能误判。实际宿主需显式 batch schema。
- 部分输入/配置边界只验证范围，未覆盖全部 NaN/类型/空 batch，不能直接当外部服务输入验证。
- 原始 OT 在某些路径上先于 padding 清理执行；padding 含 NaN 时即使 loss 能 mask 也可能污染匹配代价。跨本体/条件 OT 耦合与 prior 要单独评估。
- 原始模型 helper 并不同时证明所有 attention、gradient checkpointing、compiled、precision、prefix 输入路径已正确。

以上是需迁移时验证的边界，原源码作为来源仍完整保留。

## 精度与训练图

no_grad 不会自动禁用 dropout/BatchNorm 更新。中心差分的两侧需同一确定性场，否则随机差异会被 δ 放大。Packed student 是否与顺序执行等价，依赖没有跨 batch 的统计/混合和正确上下文打包。

时间、差分与 latent 累积使用 FP32 只是一部分；若专家已经在 BF16 中丢失小变化，最后 `.float()` 不能恢复。用真正 FP32 field 做 δ sweep/独立 oracle，再决定低精度是否可用。启用 LSD 后要监控 local-FM/LSD/gradient-norm/有效比例，不能用吞掉非有限值的办法伪造稳定。

原始三个 API 调用成本约为 `2*B_lsd + (B + 2*B_lsd)` 个样本的专家计算；若全部激活为 5B。可共享观测编码，但训练条件需要梯度时不得随意 detach。w_lsd=0 应真正在一次 B 前向后返回。
