# NaN/Inf 历史与安全边界停止

非有限值监测在已选择的数据中发现 NaN、正无穷或负无穷时，保留有限长度的前序事件摘要，并请求在下一个安全 step 边界停止。适合留下异常发生前后的上下文，避免错误继续传播后覆盖现场。

前提是调用方会在训练边界调用 `step_boundary`，并根据返回值退出循环。该功能默认关闭。以下 CPU 示例主动构造 NaN，用来演示控制流程；`/data/musaprobe` 可替换为自己的输出目录。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/nonfinite_stop",
    "mode": "tensor", "modules": ["watched"],
    "history_size": 2, "stop_on_nonfinite": True,
    "rank": 0, "world_size": 1,
}) as probe:
    for step, number in enumerate([1.0, 2.0, 3.0, float("nan")]):
        probe.set_context(step=step, phase="forward")
        probe.save(torch.tensor([number]), "watched")
        # 实际训练中，应放在该 step 的计算已经结束的位置。
        decision = probe.step_boundary(step + 1)
        if decision["stop"]:
            break

manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
trigger, = manifest["metadata"]["nonfinite_triggers"]
assert trigger["event"]["stats"]["nan_count"] == 1
assert [e["event_id"] for e in trigger["history"]] == [1, 2]
print(manifest["metadata"]["stopped_at_boundary"])
print(probe.rank_dir)
```

正常有限值不会请求停止。停止在显式边界生效；采集器卸载观测并排空已有数据，但不会主动 `exit()`、恢复权重或重新开始训练。

## 关键参数与接口

| 参数或接口 | 说明 |
|---|---|
| `history_size` | 默认 0；内存中保留的最近事件摘要上限，按事件计数，不按 step 计数。 |
| `stop_on_nonfinite` | 默认 `False`；开启后，观测到 NaN/Inf 会置停止请求。 |
| `recent_history` | 返回当前有限长度摘要的副本。 |
| `stop_requested` | 当前本地停止请求；异步 D2H 尚未完成时，统计可能还没有返回。 |
| `step_boundary(step, coordinate=…)` | 排空必要任务、协调停止并返回 `stop` 决策。 |

只开 `history_size` 而不开 `stop_on_nonfinite`，可保留触发记录而继续计算。`history_size=0` 时仍可开启停止，但不保存之前的事件摘要。

## 输出如何看

每个普通事件的 `stats` 包含 `finite_count`、`nan_count`、`posinf_count`、`neginf_count`。全为非有限值时，min/max/mean 等有限值摘要可能为 `null`；不要把它解释成零。

manifest 的 `metadata.nonfinite_triggers` 保存触发事件与之前的有限长度历史；`metadata.stopped_at_boundary` 记录实际接受停止的边界。历史是摘要，不会追溯补采早先未保存的原始张量。原始文件是否存在，仍应检查相应事件的 `state` 和 `file`。

异常数据与采集故障是两回事：一个包含 NaN 的 tensor 可以完整保存，manifest 也可以正常完成。判断数值异常看统计计数；判断保存故障看 `error`、`unsupported`、`truncated` 和存储错误记录。

## 分布式与异步边界

多 rank 必须让所有 rank 进入同一边界，并提供 `SafeBoundaryCoordinator`；任意 rank 请求停止后共同退出。缺少协调器时，不会擅自让单个 rank 独自停跑。接入方法见[安全边界热更新](hot_reload.md)。

使用 `async_d2h=True` 时，统计在后台完成，`step_boundary` 会先排空再决定是否停止。不要只轮询 `stop_requested` 就假定尚无异常。其他异步行为见[异步存储](async_storage.md)。

## 支持范围与常见错误

检测范围只包括被 step/rank/模块筛选选中、并实际取得统计的张量；[切片](tensor_slicing.md)只检测选定区域。被过滤、未运行、无法保存或未观察到的数据不能据此判为有限。

该入口不检测 GPU hang、未完成 kernel、通信死锁或所有混合精度 scaler 状态，也不提供交互式暂停后恢复。调用方忽略 `decision["stop"]` 仍会继续训练；把边界放到尚有其他 rank 等待的通信中间，也不满足安全停止前提。

[返回使用手册](../README.md)
