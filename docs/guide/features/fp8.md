# 原生 FP8 与 MT-TE 张量保存

## 作用与适用场景

FP8 适配器保存真实量化张量的原始位模式及解码所需的缩放信息，避免仅保存反量化后的 BF16/FP32 值。适合检查量化编码、逆缩放和离线数值差异。

本功能提供独立 bundle 导出，也可以接入 `PrecisionDebugger` 的标准采集和比较链路。它不恢复完整 FP8 训练状态，不读取融合 GEMM 内部未公开的中间量。

## 使用前提

原生 float8 需要具有相应 dtype 的 PyTorch。MUSA 需要 torch_musa；MT-TE 需要已兼容当前 PyTorch/MUSA 的 Transformer Engine。包安装本身不会替你安装或升级这些依赖。

示例使用固定输出名称；重复运行时请更换输出目录或采集 run ID，不覆盖已有证据。

当前 MT-TE 适配精确匹配两个类及其 `get_metadata()` 数据布局：

- `transformer_engine.pytorch.tensor.float8_tensor.Float8Tensor`：per-tensor 逆缩放。
- `transformer_engine.musa.pytorch.tensor.mtfp8_tensor.MTFP8Tensor`：rowwise 块缩放，读取实际 quantizer 的 `block_m/block_n`。

未知类、只有 columnwise 数据、未知 padding 或不匹配的块布局会拒绝。TE 原始张量切片尚不支持。

## 完整示例：采集原生 FP8

以下脚本从项目根目录运行。它只转换一个小张量，并通过显式 `save` 保存，无需已有模型。

```bash
PYTHONPATH=src python3 - <<'PY'
import torch
import torch_musa
from musaprobe import PrecisionDebugger
from musaprobe.native_fp8 import fp8_snapshot

x = torch.linspace(-2, 2, 128, device="musa").to(torch.float8_e4m3fn)
with PrecisionDebugger(
    config={"output_dir": "/data/musaprobe/fp8_capture", "run_id": "native",
            "mode": "tensor", "capture_fp8": True},
    fp8_adapter=fp8_snapshot,
) as probe:
    probe.start()
    probe.set_context(step=0, phase="forward")
    probe.save(x, "quantized_activation")
PY

PYTHONPATH=src python3 -m musaprobe inspect \
  /data/musaprobe/fp8_capture/native
```

在已有模型中使用同一构造参数，并调用 `probe.start(model)`，即可处理所选模块暴露的受支持 FP8 对象。模型边界本来只返回 BF16/FP32 时，不会因此变成原生 FP8 采集。

## 独立导出与解码

以下是完整 CPU 原生 dtype 示例；需要当前 PyTorch 支持相应转换。

```bash
PYTHONPATH=src python3 - <<'PY'
import torch
from musaprobe.native_fp8 import export_fp8_bundle, decode_fp8_bundle

x = torch.tensor([0.5, 1.0, 2.0]).to(torch.float8_e4m3fn)
export_fp8_bundle(x, "/data/musaprobe/fp8_bundle")
decoded = decode_fp8_bundle("/data/musaprobe/fp8_bundle")
torch.testing.assert_close(decoded, x.float(), atol=0, rtol=0)
print(decoded)
PY
```

从仓库运行 MT-TE 布局检查：

```bash
PYTHONPATH=src python3 examples/advanced_probe.py fp8 \
  --device musa --flavor te-tensor --output /data/musaprobe/fp8_per_tensor
PYTHONPATH=src python3 examples/advanced_probe.py fp8 \
  --device musa --flavor te-block --output /data/musaprobe/fp8_blockwise
```

`--flavor native` 则检查原生 E4M3FN。原生解码支持 E4M3FN、E5M2、E4M3FNUZ、E5M2FNUZ；这不代表每种格式在每个设备上的运算都已验证。

## 配置、输出与验收

| 项目 | 含义 |
| --- | --- |
| `capture_fp8=True` | 显式开启原始 FP8 保存；还必须传 `fp8_adapter=fp8_snapshot`。 |
| `mode="tensor"` | 保存可离线比较的载荷；FP8 原始载荷的 `stored_dtype` 为 `uint8`。 |
| `async_write=True` | 可将已固定快照的磁盘写入交给后台队列，不等同于异步设备到主机传输。 |
| `export_fp8_bundle(value, output_dir)` | 新建独立 bundle，载荷限制 64 MiB，总数组预算 128 MiB。 |
| `decode_fp8_bundle(path)` | 在 CPU 重建数值，不创建训练用 TE 对象。 |

标准采集事件包含 `fp8` 量化元数据和 `auxiliary` 附件映射。TE per-tensor 保存已有的转置数据、quantizer scale/amax；块缩放保存已有 columnwise 附件。适配器不调用 quantizer 更新状态。独立 bundle 的 `manifest.json` 记录数组 shape、dtype、SHA256 和缩放模式。

完整产品链路检查入口：

```bash
PYTHONPATH=src python3 examples/fp8_capture_check.py \
  --output /data/musaprobe/fp8_product_check
```

该脚本固定使用 MUSA，覆盖 native、per-tensor、blockwise 各自的同步和异步磁盘保存。`fp8_capture_check.json` 中每个 case 都要求：自比与重复采集通过，受控 raw 位/逆缩放改变被比较器检出，输入与量化状态未被采集改变，落盘 payload 和全部附件的 shape/dtype/SHA 与原始字段相符。查看 `persisted_state_verified`、`persisted_fields`，不能只看顶层退出码。

## 常见问题

- **没有适配器时为何 unsupported**：仅知道张量名义 dtype 不足以解释真实 FP8 位模式和缩放；工具不会静默反量化后冒充原始快照。
- **TE 类名相同但新版本失败**：布局和私有状态可能变化；检查记录的类源码指纹与实际字段，不要猜测缩放含义。
- **解码一致但训练仍不一致**：保存/解码只覆盖当前张量状态，不恢复 quantizer 的完整更新过程、随机状态或优化器。
- **能否给 TE 张量配置切片**：当前拒绝，因为 raw 数据与块缩放必须同时按正确布局切片。

[返回使用手册](../README.md)
