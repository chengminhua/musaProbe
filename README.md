# musaProbe

面向 PyTorch / MUSA 的精度调试工具，帮助你保存训练和推理中的张量、比较两次运行的数值差异，并沿模块、算子和梯度逐步定位问题。

**采集 → 比较 → 查看报告 → 缩小排查范围**

当前版本：**0.1.0**。项目展示名为 **musaProbe**，Python 包名和命令行为 `musaprobe`。

本项目独立维护，不代表所提及的第三方项目或厂商的官方产品；相关名称和适配说明不表示官方合作或背书。

[快速开始](docs/guide/quickstart.md) · [安装](docs/guide/installation.md) · [功能手册](docs/guide/README.md) · [配置说明](docs/guide/configuration.md) · [常见问题](docs/guide/faq.md)

## 主要功能

| 功能 | 可以做什么 | 使用指南 |
|---|---|---|
| 张量采集 | 保存模块输入输出、反向梯度、参数及优化器状态；按 step、rank、模块和预算控制范围 | [模块采集](docs/guide/features/capture.md)、[筛选与预算](docs/guide/features/capture_filters.md)、[梯度](docs/guide/features/parameter_gradients.md) |
| 数值比较 | 对齐两次运行，计算误差和 FP32 ULP，设置模块/API 阈值及名称映射 | [运行比较](docs/guide/features/compare.md)、[阈值规则](docs/guide/features/threshold_rules.md)、[名称映射](docs/guide/features/name_mapping.md) |
| 报告与定位 | 生成 JSON、CSV、独立 HTML 报告，查看趋势、调用栈及已观察到的数据依赖 | [报告](docs/guide/features/reports.md)、[趋势](docs/guide/features/trends.md)、[执行图](docs/guide/features/execution_graph.md) |
| 算子验证 | 重放支持的纯 ATen 调用，使用固定输入与梯度种子进行前反向预检 | [算子重放](docs/guide/features/replay.md)、[精度预检](docs/guide/features/precheck.md) |
| 配置与权重检查 | 比较训练配置、转换和比较 checkpoint，按明确布局重组参数/梯度分片 | [训练配置](docs/guide/features/training_config.md)、[checkpoint](docs/guide/features/checkpoint.md)、[分片重组](docs/guide/features/shard_reconstruction.md) |
| 采集控制 | 张量切片、步边界热更新、异常停跑、异步写盘和显式输入替换 | [采集控制手册](docs/guide/README.md#采集与控制) |
| 运行模式与框架接入 | 提供 FP8、compile、MUSA Graph、自有内核探针及训练/推理适配入口 | [运行模式](docs/guide/README.md#算子与运行模式)、[框架接入](docs/guide/README.md#框架接入) |

## 快速开始

需要 Python 3.10+。在仓库根目录安装：

```bash
python3 -m pip install .
musaprobe --version
```

已有采集数据的离线比较与报告只依赖 NumPy。以下示例还需要 PyTorch；MUSA 采集需要配套的 `torch_musa`，PyTorch/Megatron checkpoint 转换也有额外依赖。环境准备、离线安装和依赖说明见[安装指南](docs/guide/installation.md)。

```bash
python3 examples/quickstart.py --device cpu --output /data/musaprobe/demo
```

有可用 MUSA 设备时，将 `--device cpu` 改为 `--device musa`。每次运行都会生成独立的 `demo_*` 目录，其中包含：

- `comparison_normal/report.html`：同一输入、权重下的正常比较。
- `comparison_injected/report.html`：主动给首层输出增加 `0.1`，验证报告能检出差异。
- `demo_summary.json`：运行结论及报告路径。

完整步骤和报告解读见[快速开始](docs/guide/quickstart.md)。文档中的 `/data/musaprobe` 是可替换的数据目录，运行前请确认该目录可写且空间充足。

## 接入自己的模型

通过 `PrecisionDebugger` 配置采集范围，在训练循环中标记真实的 step、microbatch 和反向会话，然后离线比较参考运行与待检查运行。

建议从少量模块、少量 step 的统计采集开始；找到可疑位置后，再保存对应张量。使用[模块采集示例](docs/guide/features/capture.md)接入，按需开启 [API 采集](docs/guide/features/api_capture.md)、[反向会话](docs/guide/features/backward.md)和[优化器状态采集](docs/guide/features/optimizer_state.md)。

```bash
musaprobe compare /data/musaprobe/captures/reference_001 \
  /data/musaprobe/captures/candidate_001 \
  --output /data/musaprobe/reports/compare_001
```

上述目录需要先完成采集。`compare` 输出 `comparison.json`、`comparison.csv` 和可直接打开的 `report.html`；退出码 `0` 表示通过、`1` 表示发现差异、`2` 表示证据不完整或执行失败。

## 支持范围

常规采集面向 PyTorch eager 的可见模块和算子边界。FP8、编译、图内探针与各框架适配有各自的版本、模型和执行模式要求，详见对应功能页。

统计摘要用于筛查，逐元素判断需要完整张量。缺失、截断、未对齐和不支持的数据会明确显示；首处观测差异是排查起点，不能直接认定为根因。采集会增加同步、内存和磁盘开销，可通过筛选、切片和预算限制范围。

## 开发

```bash
MUSAPROBE_TEST_TE_RUNTIME=0 PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m pip wheel --no-deps --wheel-dir /data/musaprobe/build/wheels .
```

核心实现位于 `src/musaprobe/`，示例位于 `examples/`，回归用例位于 `tests/`。提交问题时请提供版本、最小复现代码、预期与实际结果，并检查附件中是否包含私人路径或业务数据。

使用 agent 时，从 [AGENTS.md](AGENTS.md) 选择入口：[使用 skill](skills/musaprobe-usage/SKILL.md)用于采集、比较和报告解读，[开发 skill](skills/musaprobe-development/SKILL.md)用于修改实现和验证。两个 skill 随仓库维护，可直接读取对应的 `SKILL.md`。

更多入口：[功能手册](docs/guide/README.md)、[文档目录](docs/README.md)。

## 许可证与版本

本项目采用 [Apache License 2.0](LICENSE)，版权与来源说明见 [NOTICE](NOTICE)，版本记录见 [CHANGELOG](CHANGELOG.md)。依赖组件遵循各自的许可证。

## 设计参考与致谢

本项目在精度调试功能设计和部分接口习惯上参考了 [msProbe](https://gitcode.com/Ascend/msprobe) 的公开资料与源码，并另行实现面向 PyTorch/MUSA 的采集、数据格式、离线比较和报告。感谢该项目提供的参考，参考版本记录见 [NOTICE](NOTICE)。
