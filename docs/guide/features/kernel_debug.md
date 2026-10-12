# 自有 kernel 的调试输出

## 作用与适用场景

当拥有算子源码时，可以让 kernel 将选定中间值写入独立 debug buffer，再在设备执行完成后保存和比较。本功能提供 MUSA C++ 与 Triton 两个小示例，说明如何验证这种显式输出路径。

示例计算 `y = x*x + 1`，同时保存中间值 `x*x`。它适合开发自有 kernel 时建立检查方法，不是自动读取任意设备内存或闭源融合算子内部张量的工具。

## 使用前提

- MUSA 设备与 torch_musa 可用。
- C++ 路径需要匹配的 MUSA SDK、编译器、PyTorch 扩展构建依赖。
- Triton 路径需要具有实际 MUSA 后端的 Triton，普通 CUDA 版安装不能代替。
- 源码、构建目录、缓存和输出使用当前任务可写的独立目录。

命令从项目根目录执行。没有主 CLI 的 `kernel` 子命令，入口是高级示例或 Python API。

## 运行两个完整示例

```bash
PYTHONPATH=src python3 examples/advanced_probe.py kernel \
  --device musa --backend musa \
  --output /data/musaprobe/kernel_cpp

PYTHONPATH=src python3 examples/advanced_probe.py kernel \
  --device musa --backend triton \
  --output /data/musaprobe/kernel_triton
```

这两条命令分别构建或 JIT 编译自己的小 kernel，不修改已安装的算子库。MUSA 扩展在当前设备、当前 stream 上运行，检查 dtype、连续布局、输入数量及启动错误；两种 kernel 都对越界线程进行保护。

## Python API 与参数

```python
from musaprobe.kernel_debug import run_debug_kernel

result = run_debug_kernel(
    "/data/musaprobe/kernel_custom",
    backend="musa", device="musa", elements=257,
)
print(result["status"], result.get("intermediate_max_abs"))
```

| 参数 | 含义 |
| --- | --- |
| `backend` | `musa` 为 C++ 扩展，`triton` 为 Triton JIT。 |
| `device` | 当前仅允许 `musa`。 |
| `elements` | Python API 接受 `1..4096`，默认 `257`；高级示例固定使用默认值。 |
| `output_dir` | 必须为新目录。 |

输入固定为 `[-2, 2]` 的 float32 等距数列。C++ 编译临时将 `MAX_JOBS` 设为 `2`，完成后恢复原值；其他系统配置不由本函数修改。

## 输出解读

结果目录保存 `kernel.json`、`input.npy`、`intermediate.npy` 和 `output.npy`。C++ 路径另外保存 `debug_kernel.mu` 和 `build/`。

`status=pass` 要求中间量和最终输出分别与 CPU 参考在 `atol=rtol=1e-6` 内一致；`mismatch` 表示数值不符；`unsupported` 记录编译、导入或设备执行异常。`scope=owned_square_add_kernel_explicit_debug_buffer` 和 `closed_library_internal_dump=false` 明确限定覆盖范围。

这仍是一个独立 kernel 检查。它没有自动把其他算子的中间量注入标准采集，也没有验证你的自定义 kernel 的所有输入、性能或反向梯度。

## 检查 SDK 公开头文件

如果想查找 SDK 是否暴露调试相关声明，可运行只读扫描：

```bash
PYTHONPATH=src python3 examples/advanced_probe.py kernel-inspect \
  --include-dir /usr/local/musa/include \
  --output /data/musaprobe/kernel_interfaces
```

将 `--include-dir` 换成实际 SDK 头文件目录。输出 `interfaces.json` 优先扫描 MUSA/runtime/计算库相关头文件，跳过 LLVM/Clang 实现目录和符号链接；记录扫描文件 SHA、关键词命中和截断原因。预算为最多 2000 个已扫描文件、每文件 2 MiB、总量 64 MiB，展示最多 100 条匹配。

`scanned_files`、`truncated`、`truncation` 决定证据边界。命中 `dump` 或 `debug_output` 不能证明存在可用 tensor dump ABI；未命中也不能证明厂商没有内部调试能力。该扫描始终保留 `supported_runtime_dump_verified=false`。

## 常见问题

- **扩展编译失败**：核对当前 PyTorch、torch_musa、SDK 与宿主编译工具是否匹配，先处理具体编译错误，不要将它解释为数值错误。
- **Triton 找不到设备后端**：需要支持 MUSA 的实际编译后端，工具不自动安装或替换 Triton。
- **想读 muDNN/muBLAS 内部张量**：需要厂商提供可用接口或源码级接入；当前自有 kernel 示例不能代替该能力。
- **扫描结果被截断**：缩小到明确的 runtime/计算库 include 子目录继续调查，并保留两次各自的扫描范围。

[返回使用手册](../README.md)
