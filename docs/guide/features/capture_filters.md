# 采集筛选、周期与预算

筛选决定“在哪些 step、rank、microbatch 和模块上观察”，预算决定“最多保存多少”。适合先缩小排障范围，再提高局部采集精度，避免一开始保存完整训练数据。

前提是调用方知道训练步和微批次位置，并主动设置上下文。以下是可直接运行的单进程 CPU 示例，输出路径可替换。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/filtered_capture",
    "mode": "tensor",
    "steps": [1, 3], "ranks": [0], "microbatches": [0],
    "modules": ["0"], "exclude_modules": [],
    "capture_gradients": False,
    "max_tensor_bytes": 64 * 1024,
    "max_rank_bytes": 1024 * 1024, "max_events": 40,
    "rank": 0, "world_size": 1,
}) as probe:
    probe.start(model)
    for step in range(5):
        probe.set_context(step=step, microbatch=0, phase="forward")
        model(torch.ones(2, 4))

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
assert {e["step"] for e in events} == {1, 3}
assert {e["module"] for e in events} == {"0"}
print(probe.rank_dir, len(events))
```

各筛选条件共同生效。这里另外三个 step 仍正常计算，只是不保存数据；`ranks=[0]` 也不会改变程序的分布式拓扑。

## 关键参数

| 参数 | 规则 |
|---|---|
| `steps`、`ranks`、`microbatches` | 非负整数列表；`None` 不限制，空列表不选择任何对应位置。 |
| `modules` | 默认 `["*"]`，按模块名称匹配 glob；根模块为 `<root>`。 |
| `exclude_modules` | 排除匹配名称，优先于 `modules` 的包含结果。 |
| `max_events` | 每 rank 的普通事件预算，默认 100000；耗尽时可再写一条终止 `notice`。 |
| `max_tensor_bytes` | 单张量载荷预算，默认 64 MiB；超限不保存完整张量。 |
| `max_rank_bytes` | 每 rank 张量载荷累计预算，默认 8 GiB；包括文件表示开销或异步保守预留。 |
| `stack_depth` | 记录的 Python 调用栈深度，默认 8；设为 0 关闭调用栈记录。 |

`modules=["encoder.*"]` 匹配子模块，但不包含名称恰好为 `encoder` 的模块；若两者都需要，写成 `["encoder", "encoder.*"]`。手动 `save` 使用传入的 `name` 作为模块名称。API 事件也受模块筛选约束，详见 [API 采集](api_capture.md)。

参数梯度和参数值快照忽略 microbatch 筛选，并记录 `microbatch=None`；普通模块数据及优化器状态仍按各自上下文筛选，见[参数梯度](parameter_gradients.md)与[优化器状态](optimizer_state.md)。

## 周期采集

在配置中增加以下字段，可选择 step 1、3，随后停止周期采集：

```json
{
  "monitor_every_n_steps": 2,
  "monitor_start_step": 1,
  "monitor_max_steps": 2
}
```

这是配置片段，不是完整配置文件。周期规则与 `steps` 等条件取交集；`monitor_max_steps` 限制周期位置数，不是训练的总步数。每个 step 仍需由 `set_context` 或 `step` 设置，周期采集不会自动启动或停止训练循环。

## 分布式配置

分布式运行必须让所有 rank 使用同一个显式 `run_id`，或在构造采集器前共同设置 `MUSAPROBE_RUN_ID`。rank/world size 优先使用已初始化的 `torch.distributed`，否则读取配置或 `RANK`/`WORLD_SIZE`；显式配置与已初始化进程组不一致会报错。上面的 `rank=0, world_size=1` 仅用于单进程示例，不能原样带到多 rank 作业。

仅选择部分数据 rank 时，其余 rank 仍需参与[配置或停跑协调](hot_reload.md)。

## 输出如何看与常见错误

先看 manifest 的 `states`、`event_budget_exhausted`，再看事件的 `state` 与 `reason`。超预算保存为 `truncated` 或终止通知，不能把它理解为已保存一个较小但完整的张量。需要有意保存局部元素时使用[维度切片](tensor_slicing.md)。

字节预算主要约束张量载荷，不是整个输出目录的硬配额：JSON、调用栈及应用日志也占空间；统计计算仍可能产生设备同步。统计模式不写普通 `.npy`，因此不能靠调高字节预算取得逐元素证据，应切换到 `mode="tensor"`。

没有事件时，检查 `microbatch=None` 是否被 `microbatches=[0]` 排除，以及是否把模块参数名误写成模块名。共享模块采用首个规范路径，别名不一定是可匹配的第二个模块。

[返回使用手册](../README.md)
