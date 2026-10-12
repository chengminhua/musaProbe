# 安全 step 边界热更新

热更新让正在运行的采集器在安全边界切换模块、step、预算等采集策略。适合先做粗粒度观察，发现问题后把后续采集范围缩小到目标层。

前提是调用方能明确指出上一 step 的计算已经结束。更新只改变采集配置，不改变模型、优化器或训练超参数。以下示例是单进程 CPU 程序。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/hot_reload",
    "mode": "tensor", "modules": ["first"],
    "rank": 0, "world_size": 1,
}) as probe:
    probe.set_context(step=0, phase="forward")
    probe.save(torch.ones(2), "first")
    queued = probe.request_config_update({"modules": ["second"]})
    assert queued
    # 调用方确认 step 0 的计算已经完成，再进入 step 1 边界。
    decision = probe.step_boundary(1)
    assert decision["accepted"] and decision["applied_config_version"] == 1
    probe.save(torch.ones(2), "first")   # 新策略排除它
    probe.save(torch.ones(2), "second")

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
assert [(e["module"], e["config_version"]) for e in events] == [("first", 0), ("second", 1)]
print(probe.rank_dir)
```

`request_config_update` 返回 `True` 只表示本地预检查成功；配置在边界获准后才生效。每次只保存一个待处理更新，后续请求会替换先前待处理请求。

## 关键接口

| 接口 | 说明 |
|---|---|
| `request_config_update(changes)` | 接收部分配置字典并排队，失败返回 `False`，旧配置保持有效。 |
| `request_config_reload(path)` | 读取 JSON 后提出相同的更新请求；不自动监视文件变化。 |
| `step_boundary(step, coordinate=None)` | 再次校验、协调并应用更新，设置新的 step 上下文。 |
| `decision["accepted"]` | 本次协商是否接受；还应检查 `error`、`stop`、`applied_config_version`。 |

配置文件可只包含待改字段，例如 `{"modules": ["encoder.*"]}`。当程序已经有 `probe`，且文件确实存在时，可在安全位置调用 `probe.request_config_reload("/data/musaprobe/control/update.json")`，再调用边界接口。

同一 run 内不能修改 `output_dir`、`run_id`、`rank`、`world_size`、`async_write`、`async_d2h`、`async_queue_size`。新预算不能低于已经记录或预留的量。边界会排空待完成写盘；发现存储错误时拒绝更新。

## 多 rank 协调

分布式更新必须使用真实协调器，所有 rank 按相同顺序进入边界，包含不采数据的 rank。以下是集成片段：要求训练已初始化 `torch.distributed`，各 rank 已用同一 run ID 创建并开启 `probe`，并且当前都处在同一安全边界。

```python
from musaprobe.distributed import SafeBoundaryCoordinator

with SafeBoundaryCoordinator(timeout_seconds=30) as coordinator:
    probe.request_config_update({"modules": ["encoder.*"]})
    decision = probe.step_boundary(next_step, coordinate=coordinator)
```

此片段只展示一次边界调用，`next_step` 由训练循环提供；如果 `decision["stop"]` 为 `True`，调用方必须 `break` 或返回，不再执行下一步训练。实际多步循环应复用同一个协调器，在循环外创建并关闭它。协调器使用专用 Gloo 控制组，训练可保留自己的设备通信组。各 rank 的 step、当前版本、更新内容必须一致；配置分歧保留旧配置。任意 rank 的停止请求会传给其他参与者。

默认超时为 30 秒，可设为 0～600 秒之间的正值。缺席或失联会抛 `BoundaryCoordinationError`，调用方必须结束训练循环并清理资源；失败协调器不能复用。完整独立例子见 [distributed_boundary.py](../../../examples/distributed_boundary.py)。

## 输出如何看与常见错误

事件带有 `config_version`，manifest 的 `metadata.config_versions` 保存成功切换历史。拒绝更新会产生 `kind="config_update"` 的错误说明。比较数据时保留版本身份，避免把不同采集策略的结果误当作同一覆盖范围。

不能在模块 forward 或显式 backward 会话中调用边界接口。它也不会检测外部是否仍有未完成工作；调用方要先保证安全边界成立。分布式运行只在 rank 0 调用，会导致其他 rank 缺席；把普通回调当成真实多进程协调器也不能保证一致性。

协调器不负责中断已经挂起的 GPU collective。遇到异常的停止策略见 [NaN/Inf 与边界停止](nonfinite_stop.md)。

[返回使用手册](../README.md)
