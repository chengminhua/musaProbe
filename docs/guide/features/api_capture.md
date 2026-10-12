# 高层 API 与 ATen 采集

API 采集观察选定的 `torch.*`、Tensor 方法或 `torch.nn.functional.*` 调用，以及其下可见的 ATen 重载。适合把模块层面的差异继续缩小到具体算子输入、输出，并保存有限重放需要的调用参数。

前提是 eager 执行，且在执行目标 API 的同一 Python 线程开启、关闭采集。下面的 CPU 示例不需要模型；输出目录可替换。MUSA 使用前应先导入 `torch_musa`，并把输入创建在 MUSA 上。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/api_capture",
    "mode": "tensor", "modules": ["<root>"],
    "high_level_apis": ["torch.add"],
    "apis": ["aten.add.Tensor", "aten.mul.Tensor"],
    "capture_gradients": False,
    "rank": 0, "world_size": 1,
}) as probe:
    probe.set_context(step=0, phase="forward")
    x = torch.tensor([1.0, 2.0])
    y = torch.add(x, x, alpha=2)
    z = y * 3

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
for e in events:
    print(e["kind"], e["op"], e.get("api_call_index"), e.get("api_parent"))
print(probe.rank_dir)
```

这里无需 `start(model)`。若想让算子归属某个模块，应在同一采集器中调用 `start(model)`，并执行模型前向；模块外的 API 归入 `<root>`。

## 关键参数

| 参数 | 作用 |
|---|---|
| `high_level_apis` | 高层 API 名称的 glob 列表，例如 `torch.add`、`torch.Tensor.*`、`torch.nn.functional.*`。默认关闭。 |
| `apis` | ATen 完整重载名称的 glob 列表，例如 `aten.add.Tensor`、`aten.addmm.*`。默认关闭。 |
| `capture_inputs` / `capture_outputs` | 分别控制两层 API 的输入、输出记录。 |
| `modules`、step/rank 筛选与预算 | 与模块采集共用，参见[筛选与预算](capture_filters.md)。 |
| `dependency_graph` | 额外记录已观测张量的生产与消费关系；默认关闭。 |

启用 `dependency_graph=True` 且 `apis=[]` 时，会观察 dispatch 范围内所有可见调用；先指定小范围 `apis` 更容易控制数据量。高层调用与 ATen 不是一一对应：一个调用可能对应多个、一个或没有可见的 ATen 事件。

## 输出如何看

高层张量事件的 `kind` 为 `python_api_input` / `python_api_output`，并带有 `api_level="python"`；ATen 使用 `api_input` / `api_output`。同一次 ATen 调用的输入在执行前保存，输出在执行后保存，共用 `api_call_index`。

`api_call_index` 在对应上下文和算子内计数，不是整次运行的全局序号。`api_parent` 记录包围当前调用的已观测高层 API。未观察到父调用时该字段可能不存在，不能据此猜测父子关系。

ATen 事件的 `call_spec` 保存参数结构、标量和张量输入路径；张量值仍在输入事件的文件中。未知参数对象会在结构中标为 unsupported。只有参数、输入文件及算子执行范围都满足条件时，后续重放才可执行；采集到 API 名称本身并不等于可重放。

## 支持范围与常见错误

高层模式使用 `TorchFunctionMode`，ATen 使用 `TorchDispatchMode`。它们不会自动扩散到其他 Python forward 线程，也不保证将 autograd 引擎执行的所有后向算子准确映射回前向模块。重复反向的身份应使用[显式 backward 会话](backward.md)。

编译图、设备 Graph 和融合实现内部不能按 eager API 事件覆盖范围理解。未知对象、不可保存布局及超预算事件保留明确状态；读取时检查 `state` 和 `reason`。

常见的无数据原因是只写了 `add` 而没有匹配真实 API/重载名称，或 `modules` 排除了当前执行位置。开启和关闭 dispatch 模式必须在同一线程完成；不要在工作线程中调用主线程采集器的 `close()`。

[返回使用手册](../README.md)
