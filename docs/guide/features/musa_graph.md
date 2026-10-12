# MUSA Graph 重放与图内固定探针

## 作用与适用场景

设备 Graph 将一段设备运算捕获为可重复执行的图。本功能在更新静态输入后进行真实 `MUSAGraph.replay()`，将每次输出与 eager 结果比较，适合检查重放是否读取了新输入、结果是否一致。

它不同于离线分析中的“张量依赖图”。本页处理实际设备执行；也不自动把普通 Python hook 放进设备 capture 区域。

## 使用前提

需要可用 MUSA 设备，以及提供 `torch.musa.MUSAGraph`、`torch.musa.graph` 和 stream 接口的 torch_musa。模型应可 deepcopy、输入形状固定、返回单个 Tensor。输出与编译缓存放在任务独立目录。

下面的命令在项目根目录运行：

```bash
PYTHONPATH=src python3 examples/advanced_probe.py graph \
  --device musa --output /data/musaprobe/musa_graph_boundary
```

示例创建小型 Linear + GELU eval 模型，预热三次，在独立流与当前流之间建立依赖，再 capture。两次重放分别更新静态输入，与独立 eager 模型比较。

## Python API：检查自己的模型边界

这是完整 MUSA 示例；只在确认该设备属于当前任务后运行。

```python
import torch
import torch_musa
from musaprobe.compiled import validate_execution

model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU())
result = validate_execution(
    model, (torch.arange(8, dtype=torch.float32).reshape(2, 4),),
    "/data/musaprobe/musa_graph_custom",
    device="musa", mode="graph", replays=3,
    atol=1e-5, rtol=1e-4,
)
print(result["status"])
```

| 参数 | 含义 |
| --- | --- |
| `mode="graph"` | 使用设备 Graph；不能用 CPU 代替。 |
| `device` | `musa` 或 `cuda`；后者使用真实 CUDAGraph。 |
| `replays` | Python API 为 `2..16`，默认 `2`。 |
| `atol`、`rtol` | 默认 `1e-5/1e-4`。 |
| `output_dir` | 未使用的输出目录，输入总量限制 64 MiB。 |

此函数使用模型副本和 `no_grad`，不训练原模型。每次输入相对原输入增加 `index * 0.125`。`backend` 参数只对 compile 有意义，Graph 示例无需传它。

## 观察选定的图内模型张量

当拥有模型源码时，可以将 `GraphTensorProbe` 作为子模块放在想观察的位置。它在图中执行固定 buffer 拷贝，完成 replay 后才在图外保存。完整示例：

```bash
PYTHONPATH=src python3 examples/graph_internal_check.py \
  --mode graph --device musa \
  --output /data/musaprobe/musa_graph_internal
```

需要手动接入时，下面是**已有模型上下文片段**。在 capture 前，已用 `GraphTensorProbe((2, 3), dtype=torch.float32, device="musa")` 创建 `point` 并将其注册为模型子模块，在模型 `forward()` 中调用它。此时 `debugger` 已经 `start()`，`graph` 已完成 capture；下面继续使用捕获时的**同一个 `point` 实例**，不重新构造：

```python
# Graph capture 之外：
static_inputs.copy_(new_inputs)
graph.replay()
point.save(debugger, "hidden", execution_mode="musa_graph")
```

片段中的类型来自 `from musaprobe.graph_probe import GraphTensorProbe`。每个探针保存最近一次执行的值，并记录 generation；多次 replay 后才读一次，只能得到最后一次的快照。要逐次保存，应每次 replay 完成后读取。预热和 capture 自身也可能递增 generation，因此不要假设首个重放编号为 1。

## 输出解读

边界工具的 `execution.json` 包含 `scope=eval_model_output_boundary`、每次重放的指标和状态；输入及 eager/candidate 数组分别落盘。Graph 模式 `compiled_graphs=0` 是正常情况，该字段只用于 torch.compile。

图内示例生成 `graph_probe.json` 与 `capture/rank_00000/`。标准采集事件使用 `kind=graph_tensor`、`execution_mode=musa_graph`、`graph_generation` 和显式探针名称，可用于离线检查。报告通过只说明这些固定点和执行次数满足检查。

## 支持范围与常见问题

- **CPU 模式失败**：设备 Graph 需要真实设备。CPU 的 `aot_eager` 只能用于[编译接口检查](compile.md)。
- **不能 capture 某个算子**：保留原始异常；本工具不会把失败改成 eager 成功。
- **为什么不能在 capture 中 save**：写文件、CPU 同步和 Python 回调不属于这里支持的图内观测方式。应在图内只操作预分配的设备 buffer。
- **能否观察闭源融合 kernel 的任意中间量**：不能。探针覆盖使用者显式插入的模型张量边界。
- **是否覆盖完整训练 Graph**：当前验证集中于固定 shape 的 eval 前向和选定模型张量，不承诺训练反向、优化器或任意动态图。

[返回使用手册](../README.md)
