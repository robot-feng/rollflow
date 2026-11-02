# 数据流与缓存：同步整块参考实现

## 训练

`dataset → native collator → native image/state/action preprocessing → native normalization → actions[B,H,D] + masks → native noise + grouped time[B,H] → native interpolation → full joint action expert → native velocity target/reducer → optimizer`。

G 为随机分组数，模型 batch 始终 B。仅原始 LSD 的学生分支会有明确命名的 `[LSD-, LSD+, FM]` 打包扩展；它和“把各时间组复制为不同独立 batch”的错误替代不同。

训练不读写 rolling cache。不能从另一个 episode 或另一个训练样本拿缓存作为默认条件。不能把未来 GT 动作误当 RTC 已知前缀。

## 推理

`latest observation → native preparation/prefix encoding → previous latent shifted by confirmed C → fresh native tail noise → recomputed action positions/time → one joint expert update → finite validation → commit latent → return clone(front C) → native unnormalize/action postprocess → execute C → acknowledge C`。

首次使用完整 H 噪声与 cold schedule。quick 内核采用前端逐步激活的冷启动：canonical 0=noise 的各次更新后块时间为

```
[.25, 0,   0,   0]
[.50, .25, 0,   0]
[.75, .50, .25, 0]
[1.0, .75, .50, .25]
```

warm 的输入时间为 `[.75,.50,.25,0]`，输出回到 `[1,.75,.50,.25]`。每列 16 个动作。noise-at-one 使用 `1-time` 并保留负向 Euler 增量。

这与原始源码 `_cold_iterative()` 的“同时推进到 min(final_t, level/K)”冷启动路径不同；两者分别保留，不能声称逐步同轨迹。最终阶梯/移窗对齐不等于冷启动中间路径完全相同。

## 引用执行器的边界

`RollingFlow.step` 使用调用方提供的 noise 与 predictor，不拥有模型、tokenizer、摄像头或机器人。`consumed_actions=0` 只合法用于冷启动；warm 必须为 C。返回 `[B,C,D]`，`_cache` 是实现细节；应用不得从中直接执行未去噪的尾部。

每个会话/环境应有自己的 state。多行 batch 要求稳定有序 stream IDs 且共同推进；部分 reset、行重排、异步回包、乱序 RPC、并发请求需要宿主 session manager 或每流独立 state。本内核没有线程安全承诺，没有网络幂等协议，不要让两个请求同时调用同一 state。

`context_key` 表示 episode/task/model/normalizer/frame 等兼容性身份，不是每一帧 RGB 的 hash。普通新观测更新 predictor 条件，重大身份改变应 reset。缓存提交只发生在计算成功且已确认上一块消费后；若预测函数抛异常，旧缓存保持不变。外部随机数已被抽取则不会自动回滚 RNG，重试一致性由调用方保存 noise/RNG 状态。

## 三种不同缓存

1. 任务级语言/演示缓存：仅当其真实条件不变时复用。
2. 观测 prefix KV：同一请求内去噪步骤共享；新观测重新计算。若原 forward 会向 cache 追加 suffix，必须恢复 prefix 长度。
3. 动作 latent：跨控制周期保留并移动；位置编码每次重新算。不能移动已经包含旧位置和旧噪声时间的 suffix KV/hidden。

## mask 与坐标

loss-only mask 不等于 padding attention mask，不等于 physically absent dimension，不等于 clean prefix。所有掩码的布尔方向和 reducer 分母独立核对。物理无效维度允许消除 noise；仅无监督位置可能仍是有效 attention 上下文，不能清零。

相对 EEF/关节差分动作若依赖每轮变化的 anchor，旧 latent 的数值不一定在新坐标系中成立。不能对任意有噪 latent 简单使用 clean-action 几何变换：需要明确推导先验、噪声和归一化的变换；未经验证 reset。真机执行保持原控制限幅/急停规则，不把本模型测试当安全认证。

## RTC

本包测试 clean prefix 的张量时间/冻结行为，不实现完整 RTC。独立 RTC 的 inference-delay compensation、hard prefix 和 rolling 输出队列组合必须单独建协议与闭环测试。不得把 scalar RTC guidance 时间自动广播后宣称适配任意 token-time；不得默认加入 CFG。
