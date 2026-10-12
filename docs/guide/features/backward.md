# 反向边界与重复 backward

反向采集保存模块输入、输出张量在 backward 时观察到的梯度。显式 `backward_session` 为每次反向赋予身份，适合同一图计算多个 loss、重复 backward，或需要区分重计算与普通前向的排障场景。

前提是张量参与 autograd、采集器从前向开始保持开启，直到所需反向完成。以下 CPU 示例可直接执行；普通参数累积梯度的单独快照见[参数梯度](parameter_gradients.md)。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

model = torch.nn.Linear(3, 2)
x = torch.ones(2, 3, requires_grad=True)
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/backward",
    "mode": "tensor", "modules": ["<root>"],
    "capture_gradients": True,
    "rank": 0, "world_size": 1,
}) as probe:
    probe.start(model)
    probe.set_context(step=0, microbatch=0, phase="forward")
    y = model(x)
    with probe.backward_session("loss_a", retain_graph=True):
        y.square().mean().backward(retain_graph=True)
    with probe.backward_session("loss_b"):
        y.sum().backward()

events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
grads = [e for e in events if e["kind"] in {"grad_input", "grad_output"}]
assert {e["backward_id"] for e in grads} == {"loss_a", "loss_b"}
print(probe.rank_dir)
```

两个 `retain_graph=True` 有不同职责：PyTorch 保留计算图，采集会话保留本次已经触发的观测 hook。只设置其中一个，不能得到完整的重复反向采集。采集器不会替调用方执行 backward。

## 关键接口

| 接口或参数 | 用法 |
|---|---|
| `capture_gradients=True` | 默认开启；对参与梯度计算的模块输入/输出张量安装观测 hook。 |
| `backward_session(backward_id=None, retain_graph=False)` | 显式标注一次反向；省略 ID 时自动生成非负整数。 |
| `backward_id` | 非负整数或非空字符串；同 step/microbatch/config_version 内不能重复使用。 |
| `release_graph()` | 在会话外移除全部待处理观测 hook；不释放 PyTorch 自己的计算图。 |
| `stop()` / `close()` | 清理本采集器的观测 hook；需要梯度时应在反向之后调用。 |

`capture_inputs=False` / `capture_outputs=False` 可以关闭前向张量记录，`capture_gradients=True` 仍会安装相应梯度观测。没有 `requires_grad` 或没有经过 backward 的边界，不会产生对应梯度事件。

## 输出如何看

`grad_input` 和 `grad_output` 的 `phase` 为 `backward`，保留前向的模块、step、microbatch、调用序号等定位信息，并附加当前会话的 `backward_id`。未使用会话时，该字段允许为 `null`；这不足以可靠区分同图多次反向。

这些事件是张量梯度观测，不是某个模块的唯一梯度贡献。共享输入、别名、共享模块或累积路径可能使多个边界观察到相同的梯度。参数 `.grad` / `main_grad` 快照需要另外调用[参数梯度入口](parameter_gradients.md)。

## 重计算、清理与常见错误

在显式 backward 会话中再次进入模块前向时，会记录 `recompute=True`。非重入 checkpoint 可能在取得需要的中间量后提前结束重计算，因此部分边界有重计算输入、没有完整重计算输出；manifest 的 `coverage_notes` 会说明该情况。

正常结束一个会话只移除已经触发且不再保留的 hook，其他在途微批次的观测仍保留。会话异常会释放待处理 hook，并把采集记为错误；原异常继续传播。结束任务时使用 `with` 或 `finally: probe.close()`，避免在后续训练中残留观测。

会话不支持嵌套；并发 Python 线程、任意框架调度或任意 checkpoint 策略不具有通用支持保证。`release_graph()` 会影响所有待处理观测，因此不要在仍需采集其他微批次反向时调用它。

[返回使用手册](../README.md)
