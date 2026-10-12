# 分析已观测的执行依赖

`graph` 根据采集记录中的 Tensor 生产者、输入版本和显式通信信息建立依赖图。适合沿实际观测边查看数值如何流动，以及哪些位置首次被观察到 NaN/Inf。它不同于比较报告按模块名组成的树。

分析只依赖 NumPy；生成新的 ATen 依赖证据需要 PyTorch 采集，并显式启用 `dependency_graph`。普通模块采集或主 quickstart 未开启此选项时，不能补推出完整图。

## 先生成一个小型 CPU 图

以下从仓库安装环境运行，不需要 GPU。输出路径已存在时应换新的 run_id。

```bash
python3 - <<'PY'
import torch
from musaprobe import PrecisionDebugger

class Tiny(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)

model = Tiny()
with PrecisionDebugger(config={
    'output_dir': '/data/musaprobe/manual_graph', 'run_id': 'observed_001',
    'mode': 'tensor', 'dependency_graph': True,
    'apis': ['aten.add.Tensor', 'aten.relu.default'],
    'max_tensor_bytes': 1024 * 1024, 'max_rank_bytes': 4 * 1024 * 1024,
}) as probe:
    probe.set_context(step=0, microbatch=0, phase='forward')
    probe.start(model)
    model(torch.tensor([-1., 2.]))
    probe.stop()
PY

musaprobe graph /data/musaprobe/manual_graph/observed_001 \
  --output /data/musaprobe/manual_reports/graph
```

结果保存 `graph.json` 和 `graph.html`。这个例子的外部输入没有被观察到生产者，因此预期 partial、退出 2，而不是完整全程序图。仍能查看被观察到的 add → relu 数据边。

CLI 只有采集路径和必需的 `--output`；Python API 是 `musaprobe.graph.analyze_graph(capture, output=...)`。输出目录必须位于输入之外。

## 边来自什么证据

ATen 输出的 `tensor_id` 表示已观察的 Tensor 版本，`input_tensor_ids` 指向输入；输入的 `producer_event_id` 可关联此前已保存的输出。原地操作产生新版本。缺少生产者、未观察到的修改、不能确认的别名关系保留 unknown，不按模块名、相同 shape 或相同数值猜边。

跨 rank 目前只处理显式记录的同步 `all_reduce`。所有参与 rank 必须使用相同的 `group_id`、调用 `sequence`、参与者集合和 reduction，且每个参与者都有 input/output 记录；缺一方或配置冲突为 incomplete。已有进程组中的采集程序可使用：

```python
from musaprobe.integrations.communication import observed_all_reduce

# dist 已初始化，所有 group 参与者都必须进入同一次调用。
observed_all_reduce(probe, tensor, group_id='data_parallel', sequence=0,
                    group=process_group)
```

这是实际执行通信的包装器，不是只加一个标签；默认 reduction 是 SUM，支持显式 `op`。调用方负责正确的分布式同步与唯一序号，不要在只有部分 rank 执行的分支里插入它。工具不会把普通通信日志自动转换为边。

## 如何读结果

`nodes/edges` 是图内容；`unknown_inputs` 是缺失生产者或不可信关联；`unrepresented_values` 是未表示为 Tensor 的参数；`errors/incomplete_events` 保留错误或截断。`summary` 汇总节点、边、未知输入与非有限节点数量，`coverage` 说明观测范围。

HTML 可按 rank 筛选、搜索模块/算子/Tensor ID，点击节点查看事件、邻接边和未知输入。页面最多展示 300 个匹配节点，完整数据在 JSON。筛选后未显示的边不表示原始图中没有边。

`observed_nonfinite_boundaries` 列出已观测直接上游没有非有限节点的异常节点，并保留 `has_unknown_input`。它依据采集时统计量，不重新验证全部 NPY。未采集的上游计算、未知输入或设备时序均可能影响判断，所以这些点是排查候选，不能证明根因或跨 rank 的首次真实溢出。

## 状态与支持范围

`complete` 仅表示这批已记录图证据满足完整性条件，退出 0；普通观测缺口为 partial，通信组缺失/冲突为 incomplete，均退出 2。参数处理失败也为 2。非有限节点本身不会自动把完整图改为失败状态，必须同时看 `nonfinite_nodes` 和异常边界。

它不读取融合核内部、不补全 autograd 全图、不恢复编译器 IR，也不为所有通信类型或异步通信自动建图。没有节点时先检查是否开启 `dependency_graph` 并实际命中所选 API；有 unknown 时扩大有针对性的采集范围，不要把它删除后声称全图完整。

[返回使用手册](../README.md)
