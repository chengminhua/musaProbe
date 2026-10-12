# 优化器状态采集

优化器状态采集保存动量、Adam 的一阶/二阶矩等张量，适合排查“梯度相近但参数更新开始不同”的问题。本页优先使用模型参数对象映射明确的普通 PyTorch 优化器入口。

前提是优化器已经建立状态。AdamW 通常在第一次更新后才有状态；工具不会为了采集而主动执行更新。以下 CPU 例子创建普通模型和 AdamW，完成应用自己的更新后保存状态。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger
from musaprobe.integrations.optimizer import capture_optimizer_state

model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/optimizer_state",
    "mode": "tensor", "rank": 0, "world_size": 1,
}) as probe:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    probe.set_context(step=0, microbatch=None)
    result = capture_optimizer_state(probe, model, optimizer, stage="post_step")
    print(result["status"], result["captured"])

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
print([(e["module"], e["state_stage"], e["state"]) for e in events])
print(probe.rank_dir)
```

在普通 AdamW 的两个参数上通常会看到六个张量状态：`weight` 和 `bias` 各自的 `step`、`exp_avg`、`exp_avg_sq`。具体状态键取决于实际优化器，不应据此假定所有优化器都有这六项。

## 关键接口与返回值

`capture_optimizer_state(probe, model, optimizer, stage="post_step")` 接受普通 `torch.optim.Optimizer`。`stage` 只允许 `pre_step` 或 `post_step`，它是调用方对保存时机的声明，不会改变优化器行为。

映射要求 optimizer state 的参数对象就是 `model.named_parameters()` 中的对象。发现未映射参数时返回 `unsupported`，不会按列表位置猜测 master weight 或分片参数。

| 返回 `status` | 含义 |
|---|---|
| `pass` | 本次同步采集的状态张量已成功提交为 tensor/statistics 事件；还应检查最终 manifest。 |
| `submitted` | 已提交异步任务，需排空后检查文件和状态，不能直接当作持久化成功。 |
| `skipped` | 所有已遍历状态被当前采集策略排除。 |
| `incomplete` | 有不支持或截断等情况，或没有取得可用状态。 |
| `unsupported` | 优化器类型或参数所有权不满足入口要求。 |

返回值还提供 `captured`、`skipped`、`unsupported` 明细。字符串以外的状态键、非张量状态等不会被伪装成张量数据。

## 输出如何看

事件为 `kind="optimizer_state"`、`phase="optimizer"`。`module` 使用 `<参数名>.<状态键>`，例如 `weight.exp_avg`；`optimizer_type` 和 `state_stage` 保留优化器类型、采集阶段。状态 shape、dtype、device 取实际张量值，某些优化器的步数张量可能位于 CPU，这本身不表示模型回退到了 CPU。

模块筛选需要覆盖状态名称；例如模型参数 `encoder.weight` 的动量状态可用 `modules=["encoder.*"]`。microbatch 筛选同样生效，因此 `microbatches=[0]` 会排除上下文为 `None` 的状态记录。周期记录方式见[筛选与预算](capture_filters.md)。

tensor 模式下通过事件 `file` 加载 `.npy`；statistics 模式只提供摘要。异步模式应在 `close()` 或成功 `probe.writer.flush()` 后读取，详见[异步存储](async_storage.md)。

## 通用 state_dict 入口

如果只需保留 `state_dict()` 中张量的容器路径，可使用以下片段。它依赖已存在的 `probe` 和 `optimizer`：

```python
probe.capture_optimizer_state(optimizer, name="optimizer")
```

这是另一种入口：把 state dict 作为结构化张量数据保存，使用 `optimizer` 作为模块名称。其内部参数编号不等同于稳定模型参数名，也不证明分布式所有权。两种入口各自的名称和身份不同，不应混用后直接按名称比较。

## 支持范围与常见错误

当前普通优化器入口观察已有状态，不恢复优化器，也不复制完整 checkpoint。分布式优化器、独立 master weight 或外部扁平 buffer 需要真实的参数映射适配，不能沿用普通 AdamW 的结果。

空状态往往是尚未执行第一步，或选择了不维护张量状态的配置；检查返回明细再判断。比较时应保持同一 step、同一 `state_stage`，避免把更新前后状态当成同一时刻。参数累积梯度由[参数梯度入口](parameter_gradients.md)单独保存。

[返回使用手册](../README.md)
