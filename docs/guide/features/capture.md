# 模块与手动张量采集

用 `PrecisionDebugger` 保存模型模块的输入、输出，或在代码中主动保存一个张量。适合先确认“哪一层开始出现差异”，以及观察没有独立 `nn.Module` 边界的中间值。

前提是已安装 musaProbe 和 PyTorch，程序以 eager 方式执行。本页示例可直接在 CPU 运行；在 MUSA 上使用时，先导入 `torch_musa`，再将模型和输入放到同一 MUSA 设备。`/data/musaprobe` 是可配置的输出目录，需要有写权限。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU())
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/capture",
    "mode": "tensor",
    "modules": ["0", "manual"],
    "capture_gradients": False,
    "rank": 0, "world_size": 1,
}) as probe:
    probe.start(model)
    probe.set_context(step=0, microbatch=0, phase="forward")
    output = model(torch.ones(2, 4))
    probe.save({"prediction": output, "mean": output.mean()}, "manual")

rank_dir = probe.rank_dir
events = [json.loads(line) for line in (rank_dir / "events.jsonl").read_text().splitlines()]
print(rank_dir)
print([(e["module"], e["kind"], e["tensor_path"], e["state"]) for e in events])
```

`with` 负责开始、关闭和异常清理；`start(model)` 在其中为模型安装观测 hook。示例选择 `Sequential` 的第一个子模块 `0`，再保存两份名为 `manual` 的手动数据。手动名称也参与 `modules` 筛选。

省略 `run_id` 时，单进程自动生成新目录。最终结构是 `output_dir/<run_id>/rank_00000/`。指定固定 `run_id` 时，已有 rank 目录会被拒绝覆盖。

## 关键接口与参数

| 接口或参数 | 用法 |
|---|---|
| `start(model)` | 采集模块边界；不传模型时可只使用 `save` 或 API 采集。 |
| `save(value, name, kind="output")` | 保存 Tensor 或包含张量的 tuple/list/dict；手动叶路径以 `value` 开头。 |
| `mode="statistics"` | 默认模式，只保存统计摘要；不产生普通张量 `.npy`。 |
| `mode="tensor"` | 保存统计和真实张量文件，受预算限制。 |
| `capture_inputs` / `capture_outputs` | 默认均为 `True`，控制模块及 API 输入、输出的数据记录。 |
| `capture_gradients` | 默认 `True`；本例显式关闭，只看前向。详见[反向采集](backward.md)。 |
| `set_context(step=…, microbatch=…, phase=…)` | 标注当前计算位置；采集器不会自动推断训练步。 |
| `stop()` / `close()` | 前者卸载观测，后者还排空写盘并完成 manifest；`close()` 后不能重启同一对象。 |

根模块名称是 `<root>`，其他名称来自 `model.named_modules()`。共享模块采用该遍历给出的首个路径，多次调用用 `call_index` 区分。进一步缩小范围见[筛选与预算](capture_filters.md)。

## 输出如何看

- `manifest.json`：配置、rank、结束状态、事件数和各 `state` 数量。
- `events.jsonl`：每行一个张量叶的观测，含 `step`、`module`、`kind`、`tensor_path`、shape、dtype、统计等。
- `tensors/*.npy`：仅完整 tensor 事件具有可读取文件，路径记录在事件的 `file` 中。

接上面的示例，读取一份真实张量：

```python
import numpy as np

event = next(e for e in events if e["state"] == "tensor")
array = np.load(rank_dir / event["file"], allow_pickle=False)
print(array.shape, event["dtype"], event.get("stored_dtype"))
```

BF16 普通张量以 float32 NumPy 数组保存，原 dtype 仍保留在事件中。应同时检查 manifest 和事件状态；`statistics`、`truncated`、`unsupported`、`error` 不能当作完整 tensor。统计相同也不证明逐元素相等。

## 支持范围与常见错误

模块采集观察已经暴露的张量边界，不会自动导出融合内核内部变量。常见张量容器会递归展开；标量等非张量值不会单独生成张量文件，未知对象会留下 `unsupported` 说明。

没有数据时，先检查是否调用了 `start(model)`、名称筛选是否匹配、step 是否设置正确，以及是否提前 `stop()`。需要反向数据时，应在 backward 完成后再关闭。模块观测本身不修改计算输入；[输入替换](input_replacement.md)属于另行启用的干预功能。

[返回使用手册](../README.md)
