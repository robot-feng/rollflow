# v1.0.0

- 从用户提供的 v0.5 可执行包构建，不将只有 v0.5 内容的同名 v1 文档误当成已完成的新实现。
- 新增默认 quick 与主动选择 original-LSD 的交互/非交互创建器；默认 H64/M4/C16、instant、cold4/warm1。
- 新增严格配置解析、checkpoint horizon 保护、冲突拒绝、不可静默覆盖的输出文件，以及 resolved 配置复核。
- 增加训练分组单位/单组概率选项，保持 quick 默认 Div(H)；原始 LSD 仍为 Div(H/C)。
- 瞬时/实验 LSD 参考缓存对 action_dim_mask 使用严格 bool 检查，避免 NaN/浮点掩码静默真值转换。原始归档不改。
- 新增真实小型联合注意力头的可运行示例、参数/输出/梯度/重载验证；不伪装成生产 VLA。
- 重写主 Skill 的创建体验、独立注册和验证流程；补齐数据流、缓存、原始算法局限、失败归因与宿主证据模板。
- 整理符合 Agent Skills 目录/metadata 的发布布局、跨平台 Python 测试入口和文件完整性清单。
- 原始伪代码及三份源码保持字节一致。保留原始 absolute FD 与实验 radial FD 的区分；不新增 CFG/MeanFlow/JVP 默认路线。

本版是一份完整的可移植 Skill 发布包，不代表全 XPolicy 模型、GPU、RTC异步或机器人生产验收。
