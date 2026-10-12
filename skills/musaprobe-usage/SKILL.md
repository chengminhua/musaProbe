---
name: musaprobe-usage
description: 使用 musaProbe 为模型接入张量采集、比较已有运行或 checkpoint、生成报告并解释数值差异。适用于使用工具完成精度排查；修改工具实现和开发验收使用 musaprobe-development。
---

# 使用 musaProbe 排查精度

随源码仓库使用的 skill。通过本文件位置定位仓库根目录，相对链接从本文件解析；命令从仓库根目录执行。产品名为 `musaProbe`，Python 包和 CLI 均为小写 `musaprobe`。若当前宿主没有发现该 skill，直接按仓库 `AGENTS.md` 读取本文件；不要假设它已全局安装。

## 先确定输入与运行环境

- 根据任务判断已有的是采集运行、`comparison.json`、checkpoint，还是需要接入采集的模型。已有采集直接离线分析，已有比较结果直接渲染；不要为这些任务重新启动训练。
- 确定 reference 与 candidate 的含义、实际路径、输出目录和任务指定的 Python。使用同一解释器检查版本及导入位置，避免源码和已安装旧包混用：

```bash
python3 -m musaprobe --version
python3 -c 'import musaprobe; print(musaprobe.__file__)'
```

- 从未安装的源码运行时，在相应 Python 命令前加 `PYTHONPATH=src`。安装步骤见[安装指南](../../docs/guide/installation.md)。离线分析仅需 NumPy；CPU 采集需 PyTorch，MUSA 执行需兼容的 torch_musa。不要为离线任务安装或初始化设备栈。
- 输出使用用户指定的可写目录，或仓库外的独立任务目录；不要覆盖原采集或往源码树添加实验结果。主机、账号、模型、输入数据、设备和镜像来自当前任务，不从历史验收记录推定。

## 按任务选入口

只读取与当前任务有关的页面，参数以当前 `--help`、源码及功能手册为准。

| 当前任务 | 操作与资料 |
|---|---|
| 初次体验或新模型采集 | [快速开始](../../docs/guide/quickstart.md)、[模块采集](../../docs/guide/features/capture.md)、[筛选与预算](../../docs/guide/features/capture_filters.md) |
| 比较两次采集、解读报告 | `inspect` → `compare`；[运行比较](../../docs/guide/features/compare.md)、[报告](../../docs/guide/features/reports.md) |
| 配置、命名或身份不一致 | `check-config`、`suggest-mapping`；[训练配置](../../docs/guide/features/training_config.md)、[显式映射](../../docs/guide/features/name_mapping.md)、[匹配候选](../../docs/guide/features/fuzzy_matching.md) |
| 定位反向或更新阶段差异 | [反向会话](../../docs/guide/features/backward.md)、[参数梯度](../../docs/guide/features/parameter_gradients.md)、[优化器状态](../../docs/guide/features/optimizer_state.md) |
| 权重或并行分片比较 | [checkpoint](../../docs/guide/features/checkpoint.md)、[分片重组](../../docs/guide/features/shard_reconstruction.md)；只转换可信输入，不猜测布局 |
| 重放算子、构造精度预检 | [重放](../../docs/guide/features/replay.md)、[前反向预检](../../docs/guide/features/precheck.md)；`precheck` 输入为算子名，`replay` 与 `precheck-captured` 输入为特定 rank 目录和实际 event ID |
| FP8、compile、Graph 或框架接入 | 从[功能手册](../../docs/guide/README.md)打开对应页，先核对版本、模式、模型与支持范围 |

## 已有采集的常用流程

`REFERENCE_RUN`、`CANDIDATE_RUN` 必须指向真实的运行根目录（含 `rank_XXXXX/`）或单个 rank 目录，不能是训练日志或 quickstart 的总输出父目录。`REPORT_DIR` 使用两侧输入之外的新目录。下面变量由当前任务的实际路径赋值后再执行：

```bash
python3 -m musaprobe inspect "$REFERENCE_RUN"
python3 -m musaprobe inspect "$CANDIDATE_RUN"
python3 -m musaprobe compare "$REFERENCE_RUN" "$CANDIDATE_RUN" \
  --output "$REPORT_DIR"
```

比较后读取 `$REPORT_DIR/comparison.json` 的 `overall`、`coverage`、`summary`、`first_difference` 和相关 `records`；CLI 的简要输出不含完整 records。HTML 为同目录的 `report.html`，可以直接离线打开。

`compare` 退出 `0` 为 pass，`1` 为 fail，`2` 为 incomplete 或执行失败。退出 `1` 后仍需读取已生成的报告；shell 自动化应保存退出码，避免 `set -e` 或 `&&` 阻断结果判读。其他命令按自身 JSON 状态解释，`inspect`/`report` 返回 `0` 不是精度通过。

仅重建 HTML 时使用：

```bash
python3 -m musaprobe report "$COMPARISON_JSON" --output "$REPORT_HTML"
```

这里输出是 HTML 文件路径；`inspect` 没有 `--output` 参数。`report` 不会重新比较张量。

## 新采集与设备使用

- 默认 `statistics` 模式只用于筛查；需要逐元素比较时明确使用 `tensor` 模式。先限制模块、step、rank 和字节预算，再按证据扩大范围。完整配置见[配置说明](../../docs/guide/configuration.md)。
- 两次运行分别使用独立 run_id，并核对输入、初始权重、随机状态与训练配置；同一次分布式运行的所有 rank 共用同一 run_id。按真实调度标记 step、microbatch 和 backward 会话，不凭文件名或 event_id 对齐。
- 本地演示显式传 `--device cpu`。quickstart 的 `normal=pass`、`injected=fail` 是预期正常/故障对照，脚本成功退出才表示示例检查成立。
- 只有任务确需设备或远程采集时，再读[设备实验约定](../musaprobe-development/references/device-lab.md)，沿用已获授权并复核实时资源。skill 本身不新增部署、训练、注入异常或公开数据的授权。
- 遇到框架、版本或执行模式不支持，保留原因并转向任务允许的其他路径。不得关闭 compile/Graph 或以 CPU 结果声称原模式验收通过。

## 如何给结论

- 分别说明数值差异、结构/身份错误和证据不足。`statistics_only`、截断、缺项与不支持不能当作逐元素通过；`unaligned`/`error` 可能令总体 fail，须读取具体 reason，不能直接归为模型数值缺陷。
- `verified_tensors` 包含已比较的失败项；通过数量看 `summary.statuses.pass`。只对真实观测到的模块、step、rank 和精度模式给结论。
- 不靠提高阈值、移除身份字段或补零制造通过。阈值由用户要求或明确的比较规则确定；整数/bool 和 NaN/Inf 仍按工具规则处理。
- 首处观测差异是定位线索，不等于根因；说明尚未覆盖的输入、算子或通信链路，以及下一步最小检查。
- 返回实际执行的命令、版本、退出码、关键状态和报告路径。公开分享前检查报告中的路径、调用栈及业务数据；不自动上传产物。

需要修改 musaProbe 的实现时，转到[开发 skill](../musaprobe-development/SKILL.md)；使用问题先查[常见问题](../../docs/guide/faq.md)。
