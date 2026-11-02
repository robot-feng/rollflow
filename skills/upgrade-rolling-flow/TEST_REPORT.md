# RollFlow Skill v1.0 — 本轮验证报告

## 实际执行

- 起点：用户提供的 `RollFlow_Skill_v0.5_original_preserved.zip`。修改前重新执行 **40/40** 原测试通过。
- v1 最终本地测试：`python -m pytest -q tests --junitxml=reports/pytest.xml`，**107 passed，0 failed，0 skipped**；原始日志见 `reports/pytest.log`。
- `python tools/check_originals.py`：三份原始源码 bytes/SHA256，以及 verbatim 原始 docstring 全部一致。
- Python 3.13.5、PyTorch 2.10.0+cpu、pytest 9.0.2；CUDA 不可用。确切环境见 `reports/environment.json`。未验证其他版本组合。
- 本报告不采用此前对外部 XPolicy 策略的通过数量作为本轮证据。

## 覆盖内容

| 范围 | 实际执行 |
|---|---|
| 创建与选择 | 默认回车、explicit quick/LSD、自定义 H/M/C、概率、旧 H12/H50、冲突和非法类型、JSON重算、拒绝覆盖文件 |
| 时间采样 | H64 全七种分组、Div(H)与Div(H/C)区分、native sampler、两个时间方向、可配置 group unit 和单组概率 |
| 参数/数据流 | 真正联合 attention 小头，原生 scalar→token形状适配、state-dict键/形状不变、输出/梯度对照、严格保存重载 |
| rolling | cold阶梯、warm移位/tail、cold4/warm1、消费计数、输出clone、异常不提交、身份变化、bool mask、配置变化拒绝 |
| 原始 LSD | 直接导入上传源码的 sampler/loss/step；teacher停梯度、packed batch、FM fast path、padding、微小区间过滤 |
| 独立原式 oracle | B=3中两行LSD，顺序实现对照原packed实现：loss、共享参数梯度、context梯度和调用顺序 |
| 实验 LSD | 原有 straight-path loss/方向测试继续执行，保持“实验且非原版”的标签；新残差分支初始/训练后对角恒等 |
| 其他接口 | RTC clean-prefix张量冻结、辅助token时间归属、dropout与no_grad区别、只读scanner不跟随外部symlink、未验证宿主模板 |
| 包格式 | Skill目录名/name一致、metadata版本1.0.0、相对资源链接有效、原始文件完整性 |

这些测试中的“原生 scalar”指本包明确命名的合成示例基准，不是已发布 π0/GR00T 的原生生产代码。

## 真实执行的尝鲜示例

`python examples/quickstart.py --steps 24 --output reports/quickstart.json`

- 真实小型联合注意力 head：**7,467 参数**；只扩展时间形状支持，**新增参数 0**。
- H64/M4/C16；随机种子7；24次实际 forward/backward/Adam 更新；每次 loss/梯度有限。
- 固定输入下标量与等时间矩阵输出最大差约 **2.38e-7**，原参数梯度最大差约 **8.94e-8**。
- 三轮推理专家调用数 **[4,1,1]**，每轮输出 **[1,16,3]**；reset 后需64个噪声位置重新冷启动。
- 日志保留逐步 loss；它们用于计算链路 smoke，不以 loss下降证明任务质量、LSD收益或泛化。

## 修订中实际处理的缺口

1. 原挂载的 v1 文档内容仍为 v0.5。此次从可执行 v0.5 包真正新增创建器、配置解析、示例、测试与文档，输出独立v1发布包。
2. H64、M4、C16此前只有约定，没有创建时的冲突校验和checkpoint长度保护；现在有可执行解析器和拒绝测试。
3. reference buffer 将任意数字 mask 转bool，可能吞掉 NaN/非布尔输入。现在要求bool；对两个参考执行器都做回归。
4. token-time正向测试不足以证明旧函数保持；新增真实joint attention的共享参数梯度、全G、严格重载和batch维检查。
5. 原始LSD packed测试以前主要验证调用数；新增独立顺序公式的loss/参数/context梯度对照。
6. 扫描器可能读外部文件symlink；现在明确跳过。扫描仍不是全仓兼容性判定器。
7. 原始完整性工具使用 assert，python -O 会关闭它；改为显式异常，并通过篡改文件的 -O 子进程回归。

开发期间一次临时审计命令从包根直接 import configuration 失败（模块在 assets/）；正式脚本均显式建立本包模块路径，随后使用正式入口重新验证。该临时命令不作为通过证据。

## 明确未执行

- 完整 DM05/StarVLA/π0/GR00T/Dexbotic/XPolicy 发布模型、原生数据管线与production checkpoint。
- GPU峰值显存、真正BF16/FP16 CUDA field精度、torch.compile全模型、DDP/FSDP/ZeRO。
- 全RTC/异步执行器/网络服务、真实任务数据微调、仿真成功率、真机闭环。
- universal LSD耦合时间理论正确性，或本方法一步/两步的任务质量保证。

原始 absolute-FD 算法仍有文档列出的路径/采样/执行约束；保持出处与测试通过不等于消除所有算法风险。每个目标适配必须另外填 `templates/HOST_VALIDATION.json` 并附真实证据。

## 复现

`python scripts/run_checks.py` 依次核对包hash（若清单存在）、原始源码完整性、107项测试和24步尝鲜示例。该命令不修改目标仓库、不安装系统Skill、不更新用户Library。

## 干净目录复跑

发布包已解压到独立新目录，通过 `python scripts/run_checks.py` 复跑：包hash、原始来源hash、107项pytest以及24步合成尝鲜示例均完成。随后仅更新报告并重新打包，最终包再次从新目录执行同一入口。完整最终复跑日志与ZIP一并交付。
