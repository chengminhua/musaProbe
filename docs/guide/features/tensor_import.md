# 其他框架的离线张量导入

## 作用与适用场景

当其他框架能导出 NumPy 数字数组时，可以提供显式张量名称和上下文，把这些文件导入 musaProbe 的标准采集目录，再进行检查和比较。适合跨框架数值对照，或把已有安全数组接入离线报告。

导入不运行原框架，不恢复 checkpoint，不自动推断层名映射，也不因此获得 MindSpore、TorchAir、ATB 等框架的 MUSA 执行后端。

## 使用前提

离线导入和比较只需要 musaProbe 的 NumPy 依赖。源目录包含一个 JSON 清单与其相对路径引用的 `.npy` 数组，禁止 pickle、对象数组和越界路径。

输出目录必须是新目录，且不能等于源清单目录或位于其下面。推荐使用并列的 `export/` 和 `imported/`，避免将导入结果混入原始数据。

## 完整 CPU 示例

以下用 NumPy 创建导出格式示例，不实际运行任何第三方框架。命令从源码仓库根目录执行；重复运行请换一个新的根目录。

```bash
PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("/data/musaprobe/tensor_interchange/export")
root.mkdir(parents=True, exist_ok=False)
array = np.arange(6, dtype=np.float32).reshape(2, 3)
np.save(root / "activation.npy", array, allow_pickle=False)
manifest = {
    "framework": "pytorch",
    "tensors": [{"name": "encoder.layer0.output", "file": "activation.npy",
                 "shape": [2, 3], "dtype": "float32", "step": 0,
                 "phase": "forward", "kind": "output"}],
}
(root / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

PYTHONPATH=src python3 -m musaprobe import-tensors \
  /data/musaprobe/tensor_interchange/export/bundle.json --framework pytorch \
  --output /data/musaprobe/tensor_interchange/imported

PYTHONPATH=src python3 -m musaprobe inspect \
  /data/musaprobe/tensor_interchange/imported

PYTHONPATH=src python3 -m musaprobe compare \
  /data/musaprobe/tensor_interchange/imported \
  /data/musaprobe/tensor_interchange/imported \
  --output /data/musaprobe/tensor_interchange/self_compare
```

这个自比用于验证导入链路。做跨实现比较时，把第二个目录换成独立来源导出的运行，并先核对名称、dtype、shape 和上下文；不要把两个相同输入目录当作跨框架一致性证据。

## 清单格式与参数

`import-tensors MANIFEST --framework NAME --output DIR` 的三个参数都要明确提供。支持的框架标签为 `pytorch`、`mindspore`、`torchair`、`atb`、`vllm`、`sglang`、`verl`；标签只是来源信息，CLI 的 `--framework` 是实际使用值，建议与清单保持一致。

| 字段 | 约束和默认值 |
| --- | --- |
| `tensors` | 1～10000 条记录。 |
| `name` | 非空、在整个清单中唯一，作为事件 `module` 名称。 |
| `file` | 相对清单目录的 NumPy 文件；解析后仍必须在该目录内。 |
| `shape` | 可省略并使用实际数组形状；提供时必须完全一致。 |
| `dtype` | 逻辑类型；可带 `torch.` 前缀，必须与实际存储类型匹配。 |
| `step`、`call_index` | 非负整数，默认 0。 |
| `microbatch` | 可选，未提供则为空。 |
| `phase`、`kind`、`tensor_path` | 默认分别为 `forward`、`module_output`、`$output`；由导出者提供正确语义。 |

支持普通 float16/32/64、整数和 bool。BF16 的约定是 `dtype=bfloat16`，数组实际以 float32 保存；转换应在来源框架按 BF16 值完成后进行，不能把任意 FP32 数组重新贴标签冒充 BF16。原始 FP8 及其缩放附件请使用[FP8 专用入口](fp8.md)。

单个源文件最多 64 MiB，所有引用文件合计最多 256 MiB。当前 `name` 必须全局唯一，因此该导入器不是多 step 采集日志的无损通用转换器。

## 输出解读

输出目录包含 `rank_00000/manifest.json`、`events.jsonl` 和 `tensors/`。事件为 `state=tensor`、`device=offline_import`，记录 `source_framework`；manifest 的 `metadata.backend_execution_verified=false` 明确表示没有执行来源框架。

CLI 返回 `status=complete` 表示清单和数组通过导入校验并已复制，不表示数值比较通过。后续需使用 `inspect`、`compare` 查看非有限值、对齐和误差。导入器不自动检查两个框架的张量名称是否语义等价，需要时使用手册中的显式名称映射。

## 框架发现入口

```bash
PYTHONPATH=src python3 -m musaprobe framework-info mindspore
PYTHONPATH=src python3 -m musaprobe framework-info torchair
PYTHONPATH=src python3 -m musaprobe framework-info atb
```

它检查当前环境的模块路径、发行版信息和可发现的平台插件，不启动后端，也不自动安装依赖。`available_for_probe` 只表示存在可探测入口；`missing_dependency`、`unavailable_import_path` 或 `discovery_error` 要按具体原因处理。该命令退出码为 0 也不能当作设备执行通过，必须阅读 JSON 的状态和 `backend_execution_verified`。

## 常见问题

- **把框架标签改成 musa 是否能运行**：不能；`musa` 不是框架标签，导入本身也不执行设备计算。
- **checkpoint 能否直接导入**：不能；此入口只接受声明的 NumPy 清单。模型 checkpoint 使用专门的转换/比较功能。
- **shape 或 dtype 不符**：修正导出文件或清单，工具不会通过 reshape/cast 猜测原意。
- **缺少语义名称**：先在来源框架确定张量含义；仅靠文件排序不能建立可靠对齐。
- **存在 NaN/Inf 是否导入失败**：格式正确的数字数组可以导入，后续 `inspect` 和 `compare` 会检查非有限值；`complete` 不是数值健康结论。

[返回使用手册](../README.md)
