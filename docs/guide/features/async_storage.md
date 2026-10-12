# 异步磁盘写入与 MUSA D2H

异步存储把已经固定的快照交给有界后台队列，减少采集线程直接等待文件写入的时间。适合已经缩小采集范围、需要保留真实张量的场景。队列满时会等待，不会静默丢数据。

两种模式需要分清：`async_write=True` 只把磁盘写入放到后台；额外启用 `async_d2h=True` 才使用 MUSA event 与 pinned CPU buffer 传输。两者默认关闭，不保证加速，也不等于训练计算完全不等待。

## 最小示例：异步写盘

下面可直接在 CPU 运行，输出路径可替换。每次调用 `save` 后立即复用源张量，已提交的文件仍必须保持当时的值。

```python
import json
import numpy as np
import torch
from musaprobe import PrecisionDebugger

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/async_storage",
    "mode": "tensor", "async_write": True, "async_queue_size": 2,
    "rank": 0, "world_size": 1,
}) as probe:
    value = torch.zeros(4)
    for index in range(4):
        value.fill_(index)
        probe.save(value, "mutable")
    value.fill_(-1)

# 离开 with 已关闭采集器并排空队列。
events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
for index, event in enumerate(events):
    assert event["state"] == "tensor"
    saved = np.load(probe.rank_dir / event["file"], allow_pickle=False)
    assert np.array_equal(saved, np.full(4, index))
print(probe.rank_dir)
```

仅异步写盘时，在调用线程完成统计、CPU 快照及序列化，再将不可变字节交给后台。输入可以来自 CPU 或受支持的设备，但设备到 CPU 的快照仍可能同步。

## 启用真实 MUSA 异步传输

以下为另一个完整例子，只能在有可用 MUSA 和相应 Stream/Event/pinned-memory API 的环境执行；不会回退到 CPU。

```python
import torch
import torch_musa
from musaprobe import PrecisionDebugger

if not torch.musa.is_available():
    raise RuntimeError("MUSA device is required")

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/async_musa_d2h",
    "mode": "tensor", "async_write": True, "async_d2h": True,
    "async_queue_size": 2,
    "max_tensor_bytes": 1024 * 1024, "max_rank_bytes": 16 * 1024 * 1024,
    "rank": 0, "world_size": 1,
}) as probe:
    value = torch.arange(8, device="musa", dtype=torch.float32)
    probe.save(value, "activation")
    value.zero_()

print(probe.rank_dir)
```

采集回调在当前 producer stream 上提交独立设备 clone 并记录 event；传输 stream 等待后复制到 pinned CPU buffer。后台等待传输完成，在 CPU 计算统计、序列化和落盘。槽位直到写盘完成才可复用，设备快照和 buffer 在此期间保持存活。

## 关键参数与资源

| 参数或接口 | 说明 |
|---|---|
| `async_write` | 是否启用后台磁盘队列。 |
| `async_d2h` | 是否使用 MUSA 异步设备到主机路径；必须同时启用 `async_write`。 |
| `async_queue_size` | 正整数，默认 4；控制队列容量及 D2H 槽位数。增大它也会增加缓存占用。 |
| `max_tensor_bytes` / `max_rank_bytes` | 仍然有效；D2H 对 NumPy 文件头做保守字节预留，因此接近预算时可能提前截断。 |
| `probe.writer.flush()` | 对已经开始的 writer 排空当前任务，返回是否未记录 I/O 错误；不是所有事件都完整的证明。 |
| `close()` | 排空任务、结束后台线程并写最终 manifest；推荐通过 `with` 保证调用。 |

运行过程中不能通过[热更新](hot_reload.md)切换异步模式或队列容量。选择数量少、尺寸小的张量，再评估队列容量，通常比无限增加队列更容易控制资源。

## 输出如何看

两条路径最终都保存标准事件和 `.npy`。MUSA D2H 完成的事件带有 `snapshot_transport="musa_event_d2h"`、`stats_device="cpu"`；BF16 文件仍以 float32 表示，原 dtype 保留在事件中。

异步提交尚未完成时可能是 `pending`，不要在队列排空前把缺失文件当成最终结果。完成后检查 manifest 的 `status`、`metadata.storage_errors` 以及每条事件的 `state`、`file`。进程退出码 0 或 `flush()` 返回 True，不能替代这些完整性检查。

## 支持范围与常见错误

MUSA D2H 仅接受普通 dense MUSA Tensor/Parameter。CPU、任意 Tensor subclass、原始 FP8、稀疏、量化和复数张量不支持，不会静默切成同步路径后继续冒充异步。普通异步磁盘写入仍可与受支持的独立 FP8 快照适配器配合。

非连续张量可先固定为连续设备快照；梯度 hook 可在 autograd 引擎回调线程提交。它不代表自动采集任意 Python forward 线程中的 API，参见 [API 采集](api_capture.md)。

原地修改必须遵守训练程序自己的 stream 依赖；采集器不能修复原本就存在的跨 stream 数据竞争。异常 I/O 会记录错误，但无法保证故障文件系统上的最大等待时间，Python 也不能安全强杀阻塞中的写盘线程。正确性检查不等于证明传输与计算实现了有效重叠。

[返回使用手册](../README.md)
