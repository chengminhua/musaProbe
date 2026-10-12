# 按维度切片采集

切片采集只保存张量指定区域，并保留原始形状和实际选择范围。适合大激活中只检查某些 batch、token、通道，或先用小范围数据定位问题。

切片作用于保存的数据，不改变模型收到的张量。前提是知道张量维数和目标事件路径；可先用[模块采集](capture.md)查看 `module`、`kind`、`tensor_path`。以下 CPU 示例可直接运行。

## 最小示例

```python
import json
import numpy as np
import torch
from musaprobe import PrecisionDebugger

x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
original = x.clone()
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/tensor_slicing",
    "mode": "tensor", "rank": 0, "world_size": 1,
    "tensor_slices": [{
        "module": "activation", "kind": "output", "tensor_path": "value",
        "selection": [[1, 4, 2], [1, None, 1]],
    }],
}) as probe:
    probe.save(x, "activation")

event, = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
saved = np.load(probe.rank_dir / event["file"], allow_pickle=False)
assert torch.equal(x, original)
assert np.array_equal(saved, x[1:4:2, 1:].numpy())
print(saved.shape, event["tensor_slice"])
print(probe.rank_dir)
```

保存结果为形状 `[2, 4]`，对应原张量第 1、3 行及第 1 列之后的数据。没有匹配切片规则的事件按原形状处理。

## 关键参数

`tensor_slices` 是规则列表，每条规则必须包含且只包含以下四项：

| 字段 | 说明 |
|---|---|
| `module` | 事件模块或手动 `save` 名称，支持 glob。 |
| `kind` | 例如 `input`、`output`、`api_input`、`parameter_grad`，支持 glob。 |
| `tensor_path` | 模块输入常为 `args[0]`，手动保存单张量为 `value`；可精确匹配或使用 glob。 |
| `selection` | 每个轴一个 `[start, stop, step]`，轴数必须等于张量维数。 |

Python 中省略项写 `None`，JSON 中写 `null`。步长只能是正整数或省略；负起止索引按该轴长度规范化，越界起止按 Python slice 规则处理。当前不支持负步长、布尔掩码或高级索引。

匹配路径时可直接填写完整的 `args[0]`，不必为方括号做 glob 转义。多个规则同时匹配同一张量会产生错误事件，因此重叠规则不能用来对同一张量连续做两次切片。

## 输出如何看

事件的 `shape`、统计和 `.npy` 都描述切片。`tensor_slice` 记录：

```json
{
  "original_shape": [4, 5],
  "selection": [
    {"start": 1, "stop": 4, "step": 2},
    {"start": 1, "stop": 5, "step": 1}
  ]
}
```

未切片事件不会填写这个字段。比较两份切片时必须核对原形状及全部规范化范围，不能仅凭保存后的 shape 相同就认为位置相同。

切片在张量载荷预算判断前生效；它会改变统计覆盖范围，[NaN/Inf 检测](nonfinite_stop.md)也只看到保存区域。未保存区域是否异常仍然未知。

## 支持范围与常见错误

普通可保存张量支持这类轴切片。TE FP8 的原始数据依赖量化尺度布局，当前拒绝切片；切片也不能作为完整参数或梯度的分片重组证据。

轴数不符、重复匹配等运行时问题会记录 `error`，不能当成空切片成功。单个 tensor 太大却没有切片记录时，检查手动保存路径是否写成了模块输入的 `args[0]`，或 `kind` 是否与实际事件一致。预算超限仍可能产生 `truncated`，参见[筛选与预算](capture_filters.md)。

[返回使用手册](../README.md)
