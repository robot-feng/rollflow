# 来源和本轮范围

第一来源为用户上传的三个原始代码文件及其中的完整算法 docstring。对应原始 bytes/hash 见 `original_manifest.json`；本包实际执行其纯 PyTorch 核心，不重写成另一份类似 loss。

v1 继承了用户提供的 v0.5 archive 的 portable reference assets 和 40 个测试。继承不代表历史报告自动成为本轮结果；本轮先重新执行基线，再加入创建选择和真实小型联合注意力示例及回归用例。

本轮查阅的公开规范：

- Agent Skills Specification： https://agentskills.io/specification 。目录名匹配 name，版本放 metadata，使用相对资源路径，详细材料分离到 references。本包不宣称某个 agent 产品已经安装或运行此 Skill。
- PyTorch broadcasting semantics： https://docs.pytorch.org/docs/stable/notes/broadcasting.html 。用于核对形状扩展概念；运行版本见实际测试报告。
- PyTorch no_grad： https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html 。教师停梯度与 train/eval 的控制分别核对；不能用 no_grad 替代 deterministic field 的检测。

这些网页的 stable 文档可能对应比本机更新的版本。可执行结果以本轮 Python/PyTorch/pytest 环境和日志为准。

没有在本轮新拉取/训练全套 XPolicy、π、GR00T、Dexbotic 或 LeRobot；未将往轮源码清点数量转为本包支持数量。
