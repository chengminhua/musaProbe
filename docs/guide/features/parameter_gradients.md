# 参数梯度与参数值快照

`capture_parameter_grads` 在调用方选定的位置读取每个可训练参数当前的梯度。适合检查梯度累积、归约、缩放或裁剪前后的差异。它与模块的 `grad_input` / `grad_output` 观测不同，后者见[反向采集](backward.md)。

前提是 backward 已产生梯度，并且调用方知道此刻梯度处于哪个阶段。以下 CPU 示例在普通 SGD 更新前采集，不改变梯度或执行额外的 optimizer step。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

model = torch.nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/parameter_gradients",
    "mode": "tensor", "capture_gradients": True,
    "rank": 0, "world_size": 1,
}) as probe:
    probe.set_context(step=0, phase="forward")
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    probe.capture_parameter_grads(model)
    optimizer.step()

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
for e in events:
    print(e["parameter_name"], e["gradient_source"], e["gradient_scope"], e["state"])
print(probe.rank_dir)
```

这里没有安装模块 hook，只在显式调用处保存参数梯度。若之前 `start(model)` 绑定了一个模型，可以省略 `capture_parameter_grads` 的 `model` 参数；模型不明确时应显式传入。

## 关键行为与参数

| 项目 | 行为 |
|---|---|
| `capture_gradients` | 必须为 `True`；设为 `False` 时该入口直接跳过。 |
| 可训练参数 | 只处理 `requires_grad=True` 的参数。 |
| 梯度来源 | 未提供 resolver 时，优先读取非空 `parameter.main_grad`，否则读取 `.grad`。 |
| 筛选 | 服从 step、rank、模块和周期规则；参数所属模块参与匹配，根参数属于 `<root>`。 |
| microbatch | 参数累积梯度忽略 microbatch 筛选，并记录 `microbatch=None`。 |
| 调用位置 | 由调用方决定；工具不自动执行归约、unscale、clip、zero_grad 或 optimizer step。 |

例如参数名 `encoder.linear.weight` 的模块筛选名称是 `encoder.linear`，不是完整参数名。需要减少数据量时调整[筛选与预算](capture_filters.md)。

## 输出如何看

事件使用 `kind="parameter_grad"`、`phase="optimizer"`、`aggregation="accumulated"`，`parameter_name` 是模型参数名。`phase` 表示采集位置的分类，不表示工具替你执行了优化器操作。

通用入口记录 `gradient_scope="full_parameter"`、`gradient_source="grad"` 或 `"main_grad"`，并把阶段标为 `caller_boundary_unspecified`。这个标记不证明分布式归约已经完成。要比较两份梯度，必须确认它们来自相同的累积、缩放和裁剪阶段。

如果参数在该位置没有梯度，记录显式 `unsupported` 和原因；不会把缺梯度当作零张量。保存的 `shape` 描述实际梯度张量，`original_parameter_shape` 保留模型参数形状。

## 分布式梯度的明确归属

分片优化器需要能证明实际拥有关系的 resolver。以下仅是集成调用片段，要求 `model`、`probe` 和经过验证的 `resolver` 已存在：

```python
probe.capture_parameter_grads(model, gradient_resolver=resolver)
```

resolver 的签名为 `resolver(name, parameter)`，返回包含 `tensor`、`state`、可选 `reason` 和 `metadata` 的字典。`state` 只能是 `tensor`、`unsupported`、`error`。允许的梯度元数据是：

- `gradient_scope`：`full_parameter`、`optimizer_owned_shard` 或 `not_owned_by_rank`。
- `original_parameter_shape`、`parameter_flat_range=[start, end]`：原形状及拥有的扁平半开区间。
- `gradient_stage`、`gradient_source`：真实采集阶段和来源。
- `gradient_parallel_coordinates`：实际并行坐标；保存 owned shard 时必须非空。

成功返回分片张量必须同时给出范围、阶段、来源和并行坐标，张量元素数必须等于区间长度。未拥有区域应返回 `tensor=None` 与明确原因，不会退回读取整块 `main_grad`。该接口不靠张量顺序猜测优化器归属，也不允许 resolver 覆盖事件身份或核心张量元数据。

## 参数值与常见错误

需要额外观察权重本身时，设置 `capture_parameters=True` 并 `start(model)`。它产生 `kind="parameter"`；在被选 step 的开始/模块调用位置，每个参数每 step 最多记录一次，不能把它当作每次 optimizer step 后的自动权重快照。

常见错误是先 `zero_grad(set_to_none=True)` 再采集，或者把尚未归约的 `main_grad` 当成全局梯度。普通完整参数快照不保证适配任意分布式优化器；master weight、flatten buffer 和分片布局必须由相应集成提供真实映射。

[返回使用手册](../README.md)
