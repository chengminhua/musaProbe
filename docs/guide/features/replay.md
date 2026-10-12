# 单次 ATen 调用重放

## 作用与适用场景

`replay` 从采集文件恢复某一次 ATen 调用的输入、标量参数和容器结构，在指定设备重新执行，再与当时保存的输出比较。适合将已定位的数值差异缩小到一个算子，或用 CPU 检查同一组输入。

它重放单次纯算子调用，不恢复训练进度、模型、优化器、随机状态或完整反向传播。参数梯度分片不能作为完整参数直接重放。

## 使用前提

- 当前 Python 环境安装 PyTorch；选 `musa` 时需要可用的 torch_musa，选 `cuda` 时需要真正可用的 CUDA 后端。
- 输入是一个 `rank_XXXXX` 目录，其中有 `events.jsonl` 和实际 NumPy 张量文件。
- 调用的所有输入和唯一输出均为 `state=tensor`，且保存了 `call_spec`。统计模式、截断数据和仅模块边界采集不能替代完整 API 采集。

下面的命令从项目根目录执行，以 `PYTHONPATH=src` 使用当前源码。若已安装包，也可以去掉该前缀。

## 完整 CPU 示例：先采集，再重放

选择一个尚未使用的 run ID；已有采集目录不会被覆盖。

```bash
PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path
import torch
from musaprobe import PrecisionDebugger
from musaprobe.replay import replay_call

class AddModel(torch.nn.Module):
    def forward(self, left, right):
        return torch.add(left, right, alpha=2)

root = Path("/data/musaprobe/replay_demo")
model = AddModel()
probe = PrecisionDebugger(config={
    "output_dir": str(root), "run_id": "capture", "mode": "tensor",
    "apis": ["aten.add.Tensor"], "capture_gradients": False,
})
with probe:
    probe.start(model)
    probe.set_context(step=0, microbatch=0, phase="forward")
    model(torch.tensor([1., 2.]), torch.tensor([3., 4.]))

rank = root / "capture" / "rank_00000"
events = [json.loads(line) for line in (rank / "events.jsonl").read_text().splitlines()]
event = next(e for e in events if e["kind"] == "api_output" and e["op"] == "aten.add.Tensor")
print("event_id:", event["event_id"])
print(json.dumps(replay_call(rank, event["event_id"], device="cpu"), indent=2))
PY
```

也可以使用 CLI。将 `EVENT_ID` 改成上一步打印的真实编号；不要把事件在文件中的行号当作编号。

```bash
EVENT_ID=3
PYTHONPATH=src python3 -m musaprobe replay \
  /data/musaprobe/replay_demo/capture/rank_00000 \
  --event-id "$EVENT_ID" --device cpu \
  --output /data/musaprobe/replay_demo/replay.json
```

## 参数

| 参数 | 含义 |
| --- | --- |
| `rank_dir` | 单个 rank 的采集目录。 |
| `--event-id` | 同一次调用的一个 `api_input` 或 `api_output` 事件编号，必填。 |
| `--device` | `cpu`、`musa`、`cuda`，默认 `cpu`。 |
| `--reference-dtype` | `same` 保留采集时逻辑 dtype；`float32` 将浮点输入升为 FP32，默认 `same`。 |
| `--atol`、`--rtol` | 非负有限容差。缺省时 FP32/FP64 为 `1e-5/1e-4`，FP16 为 `1e-3/1e-3`，BF16 为 `1e-2/1e-2`。 |
| `--output` | 可选 JSON **文件**路径；不是保存重放张量的目录。 |

FP32 参考仍与原来保存的输出比较，因此升精度后的结果不同并不自动证明原算子错误。

## 输出解读

标准输出和可选 JSON 包含 `status`、`op`、`replay_scope`、容差以及误差指标。`pass` 表示本次重放的有限值在容差内一致；`mismatch` 表示数值、shape 或 dtype 不符；`unsupported` 表示无法按当前范围重放；`error` 表示环境等必要条件失败。非有限值不会因为两侧都出现就判为通过。

CLI 退出码：通过为 `0`，数值不一致为 `1`，不支持或执行错误为 `2`。该入口不保存新输出张量；需要前后向数组证据时看[算子预检](precheck.md)。

## 支持范围与常见问题

白名单为 `aten.` 下的 `add.Tensor`、`sub.Tensor`、`mul.Tensor`、`div.Tensor`、`mm.default`、`bmm.default`、`matmul.default`、`relu.default`、`sigmoid.default`、`tanh.default`、`sin.default`、`cos.default`、`pow.Tensor_Scalar`、`linear.default`、`gelu.default`、`layer_norm.default`、`softmax.int`。随机、原地修改、通信和自定义 kernel 不在范围内。

- **提示没有 call specification**：原采集未记录完整标量/容器参数，需重新进行 API 张量采集，不能只补猜测的参数。
- **找不到白名单中的 `linear`**：实际 dispatch 可能分解为 `addmm` 等其他重载，以事件的 `op` 为准；Python 函数名不保证对应同名 ATen 事件。
- **统计文件不能重放**：均值和最大值不足以恢复输入，需要 tensor 模式并避免预算截断。
- **换设备后不一致**：先核对 dtype、容差、形状和算子实现差异；单算子结果不能直接归因于硬件故障。

[返回使用手册](../README.md)
