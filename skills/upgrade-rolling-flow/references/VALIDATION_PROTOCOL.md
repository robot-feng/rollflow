# 验证、失败归因和验收

## 配置不是模型适配

`configure.py` 通过只表示参数内部一致。配置中的 `integration_status` 始终为 `configuration_only_not_adapted`；宿主验证写在独立报告，不能把一个配置布尔值改成“部署通过”。

创建流程需要测试默认回车、明确自定义、合法非 2 次幂 H、旧 checkpoint、H/C/M 冲突、非法概率/浮点整数、LSD 主动选择、文件冲突不覆盖、派生字段一致。每次变更默认值后同步更新样例与使用说明。

## 五条证据链

| 链路 | 必须核对 |
|---|---|
| 原生等价 | 固定 noise/time/输入，G1 的 output/loss/全部共享参数和条件梯度；原生 dropout 状态，必要时固定随机性 |
| 真正 token-time | 所有 G 都真实前后向；模型 batch 不因 G 增大；每层和最后层时间形状正确；错形状必须失败 |
| 模型生命周期 | 原版可并行导入、独立注册、strict checkpoint、保存重载、初始化、optimizer新参数/恢复、EMA（宿主已有才处理） |
| rolling | 新观测更新、C 消费、cold/warm预算、移窗/tail、prefix/suffix KV、reset、身份变化、mask、输出别名、异常回滚 |
| 质量/成本 | 训练与生成分别报告；真实数据 heldout、同 checkpoint fresh/rolling、连续 warm、突变、1/2/4/8预算、实际 NFE/FLOPs/显存/延迟 |

## 可选 LSD 独立增加

保留原始 pseudocode/源文件/采样配方；记录每次 field 调用的 batch、grad_enabled、source/target 和 context。检查教师两次均停梯度，学生 local-FM 不丢样本，`B_lsd` 子集和 packed顺序正确。整向量对角、有效/无效区间、边界、两种时间方向映射、determinism、差分精度、mask 分母和训练 curriculum 分开验证。

对 coupled action field 使用独立解析反例检查 endpoint stencil 语义，不把原式和实验直线路径混为一谈。local FM 非零是重要监督锚点，关闭它属于另一个实验。近边界大量样本被过滤会改变有效目标分布，需要记比例而非只看总 loss。

## 故障注入

至少注入：符号反转；最后调制仍用标量；把 H 变模型 batch；非法 cold budget；未消费即移窗；旧观测 prefix；返回值原地修改；有效 velocity NaN；inactive velocity NaN；session/mask/model ID 改变；丢失新参数；参数名相同但函数 schema 不同。

一次负例被测试“检测出来”只能说明检测器对这次负例有效，不是所有错误都排除。遇到预期拒绝应写 expected_rejection，不可冒称训练成功。

## 证据等级和状态

继承 `MIGRATION_CONTRACT.md` 的 A源码/B方法/C小型原生头/D发布权重/E仿真/F真机等级；额外将本包自建 toy 标为 `synthetic_reference`，不能写成 C原生模型。

每个真实策略分支独立记录 PASS/FAIL/BLOCKED/NOT_RUN/EXPECTED_REJECTION、命令、环境、commit/hash、测试输入、scope、是否替身。相同 head 的证据可以引用，但共享 head 不代表外层 processor/训练/执行队列都相同。

## 失败归因与迭代

- Skill 规则遗漏：修改通用规范，并加入通用反例和至少第二个实现验证。
- 宿主特例实现错误：只改该 adapter，保留最小 reproduction 和原生对照。
- 测试/报告错误：修复断言或聚合器，用实际日志重新生成；不手工改 pass 数。
- 算法假设失效：记录反例/适用域，不能把它藏成工程 bug，不能宣称换超参一定解决。

最后在干净目录重放补丁，再运行所宣称的测试。不得复用修改之前的日志作为修改后版本证明。代码规范：公共接口类型与形状文档、中文解释关键差异、许可证保留、无绝对开发机路径、无 import 时注册覆盖、无不必要重构。
