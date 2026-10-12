# 算子前向与反向预检

## 作用与适用场景

预检用受控的小输入检查一个算子的前向输出和向量雅可比积（VJP）。目标设备的结果与使用相同可表示输入、相同梯度种子的 CPU float64 参考结果比较，适合确认环境基本算子能力、定位精度差异，或复查已有采集输入。

`precheck` 生成合成输入；`precheck-captured` 读取真实采集输入，但仍生成新的梯度种子。两者都不恢复原训练的上游梯度、随机状态或优化器。

## 使用前提

安装 PyTorch；MUSA/CUDA 运行还要求对应后端和可用设备。输入必须是普通稠密非 FP8 浮点张量。输出目录应尚不存在，所有命令在项目根目录运行。

## 生成输入并检查前后向

```bash
PYTHONPATH=src python3 -m musaprobe precheck aten.linear.default \
  --device cpu --dtype float32 --seed 42 \
  --output /data/musaprobe/precheck_linear_cpu

PYTHONPATH=src python3 -m musaprobe precheck aten.linear.default \
  --device musa --dtype float32 --seed 42 \
  --output /data/musaprobe/precheck_linear_musa
```

独立 CPU 生成器产生输入，不改变调用方的全局 CPU RNG。`seed` 控制输入，实际 VJP 种子使用 `seed + 1`，并写入结果。

从源码仓库运行全部白名单算子的独立示例：

```bash
PYTHONPATH=src python3 examples/advanced_probe.py precheck \
  --op all --device musa --dtype float32 \
  --output /data/musaprobe/precheck_all
```

该示例使用内置容差。需要改变容差时，对单个算子使用主 CLI 的 `--atol`、`--rtol`。

## 使用真实采集输入

以下片段要求已有完整的 API 张量采集。先从 `events.jsonl` 选择白名单调用的真实 `event_id`；采集方法见[调用重放](replay.md)。

```bash
EVENT_ID=3
PYTHONPATH=src python3 -m musaprobe precheck-captured \
  /data/musaprobe/replay_demo/capture/rank_00000 \
  --event-id "$EVENT_ID" --device musa --gradient-seed 42 \
  --output /data/musaprobe/precheck_captured
```

它使用原输入及 `call_spec`，重新计算 CPU 参考与目标设备结果；**不将原来记录的输出作为比较目标**。要与原输出比较，使用 `replay`。

## 参数

| 入口/参数 | 含义 |
| --- | --- |
| `precheck OP` | 完整 ATen 重载名称，不能只写 `linear`。 |
| `--device` | `cpu`、`musa`、`cuda`，默认 `cpu`。 |
| `precheck --dtype` | CLI 接受 `float32`、`float16`、`bfloat16`；默认 FP32。 |
| `precheck --seed` | 合成输入种子，默认 `0`。 |
| `precheck --atol/--rtol` | 默认 `1e-5/1e-4`，不因低精度自动放宽。 |
| `precheck-captured --event-id` | 必填，指定实际采集调用。 |
| `precheck-captured --gradient-seed` | 新 VJP 种子，默认 `0`。 |
| `--forward-only` | 两个入口均可使用；跳过 VJP。 |
| `--output` | 新建结果目录，必填。 |

`precheck-captured` 的 CLI 没有容差选项。需要显式容差时，Python API 为 `precheck_captured(rank_dir, event_id, output_dir, device="musa", gradient_seed=42, atol=..., rtol=...)`。`run_precheck` Python API 和高级示例还接受 `float64`，不应将它误写成主 CLI 的可选值。

## 输出解读

单算子目录包含 `precheck.json`、`input_N.npy`、`reference_output.npy`、`candidate_output.npy`。执行 VJP 时另有 `gradient_seed.npy` 和两侧各输入梯度文件，确切文件名以 JSON 的 `files` 映射为准。数组按 float64 保存，原输入逻辑 dtype 记录在 `input_dtypes`。

`checks` 分别记录前向与各输入梯度的误差；`status=pass` 要求所有检查均通过且没有非有限值。`provenance.kind` 区分 `synthetic` 与 `captured_inputs`。真实输入预检明确记录 `captured_output_checked=false` 和 `original_training_backward_replayed=false`。

全部算子示例另外生成 `summary.json` 和编号子目录；有任一项未通过时总状态为 `incomplete`，需打开对应 case 看原因。

## 支持范围与常见问题

支持与[调用重放](replay.md)相同的 17 个纯 ATen 重载，包括 Linear、GELU、LayerNorm 和 softmax。输入总量和输出按 float64 参考大小各限制为 64 MiB；先使用 meta 运算检查输出膨胀。

- **BF16 比较失败**：低精度误差可能超过默认严格阈值。先检查逐项指标，再依据用途显式设定阈值；工具不会为了通过自动放宽。
- **设备不支持 float64**：CPU 参考固定为 float64；目标设备使用原输入 dtype，通常从 FP32 开始。
- **想验证真实训练反向**：这里的 VJP 使用新种子，只验证当前算子的局部导数；需要原训练梯度时应在实际反向中采集。
- **CPU 参考能否作为数学真值**：它是较高精度的工程参考，不保证所有实现都逐位一致。

[返回使用手册](../README.md)
