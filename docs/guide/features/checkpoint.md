# 转换并比较 checkpoint 张量

这个功能将选定的 checkpoint tensor 字典转换成安全 NumPy bundle，再按完整参数名比较。适合检查训练前后权重、两份模型权重或已明确布局的分片结果是否一致；它不负责加载模型执行，也不恢复训练。

比较 bundle 只需要 NumPy。读取 `.pt` 或原生 `torch_dist` 的**转换步骤**才懒加载 PyTorch。请保留原始 checkpoint，输出目录放在输入之外；转换要求新目录或空目录。

## 无 PyTorch 的最小比较例子

下面创建两个合成 bundle，一个参数中的最后一个值不同。目录已存在时拒绝覆盖。

```bash
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/data/musaprobe/manual_checkpoint')
root.mkdir(parents=True, exist_ok=False)
for name, delta in [('reference', 0.), ('candidate', 1.)]:
    folder = root / name
    folder.mkdir()
    value = np.arange(4, dtype=np.float32).reshape(2, 2)
    value[-1, -1] += delta
    np.save(folder / 'weight.npy', value, allow_pickle=False)
    manifest = {'format': 'musaprobe_checkpoint', 'schema_version': 1,
                'status': 'complete', 'parameters': [
                    {'name': 'projection.weight', 'file': 'weight.npy',
                     'shape': [2, 2], 'dtype': 'torch.float32'}]}
    (folder / 'checkpoint.json').write_text(json.dumps(manifest))
PY

# 预期 fail / 退出 1，仍生成报告。
musaprobe compare-checkpoint /data/musaprobe/manual_checkpoint/reference \
  /data/musaprobe/manual_checkpoint/candidate \
  --output /data/musaprobe/manual_reports/checkpoint
```

输入可为 bundle 目录或其中的 `checkpoint.json`。每个文件引用相对该清单，必须留在 bundle 内；绝对路径、`..`、逃逸符号链接、对象数组或元数据与实际 shape/dtype 不符都不能通过。

## 转换已有 PyTorch checkpoint

把下面命令的输入替换为已有 checkpoint。若暂时没有文件，可先用 CPU PyTorch 创建一个很小的教学样例；这不代表真实模型训练：

```bash
python3 - <<'PY'
from pathlib import Path
import torch

folder = Path('/data/musaprobe/checkpoints')
folder.mkdir(parents=True, exist_ok=True)
with (folder / 'model.pt').open('xb') as stream:
    torch.save({'model': {'projection.weight': torch.arange(4.).reshape(2, 2)}}, stream)
PY
```

```bash
musaprobe convert-checkpoint /data/musaprobe/checkpoints/model.pt \
  --key model --max-bytes 1073741824 \
  --output /data/musaprobe/checkpoint_bundles/model
```

`--key` 选择点分字典路径，例如 `model`；不是参数名前缀。省略时自动识别唯一的 `model` 或 `state_dict` 字典；两者都存在必须显式选择。选中的字典应是参数名到 Tensor 的映射，其他条目记为 unsupported，不会当作权重成功加载。

转换使用 `torch.load(..., map_location='cpu', weights_only=True)`，不会因反序列化失败回退到任意 pickle 执行。`--max-bytes` 默认为 1 GiB，约束源文件、解压后载荷以及转换后 tensor 总量；BF16 保存成 FP32，预算按保存后的大小计算，逻辑 dtype 保留。

## 原生分布式目录与显式布局

含 `.metadata` 的 `torch_dist` 目录可以作为转换输入；此时 `--key` 是**存储 key 前缀**，与 `.pt` 的点分路径含义不同。转换器核对普通矩形 chunk 的全局形状、offsets/sizes、完整覆盖和安全文件范围，并以 `weights_only=True` 读取各载荷。单 chunk 默认限制 64 MiB，CLI 没有单独调整该限制的选项。

已知 MCore `MCoreSavePlan` 缓存结构可受限解码成本地惰性对象，无需导入 Megatron。N-D flattened/reformulated tensor、未知布局、Zarr 目录及 legacy rank 目录的自动组装尚不支持。legacy 单个 `.pt` 只有能按 weights-only 读取并明确选中 tensor 字典时才可单独转换；不会猜测 TP/PP 轴。

对于明确的连续扁平分片，`.pt` 转换的 `--layout` 可提供：

```json
{"parameters": {"local.weight": {
  "name": "projection.weight",
  "format": "megatron_contiguous_flat_v1",
  "original_shape": [2, 2], "flat_range": [0, 2],
  "parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 0}
}}}
```

这是调用者提供的物理布局声明，不是自动识别的模型规则。bundle 内同名参数的所有分片必须共同覆盖完整参数，无重叠、缺口且坐标/shape/dtype 合法；单独 `[0,2)` 的半个参数不能比较通过。原生 `torch_dist` 目录模式不能同时传 `--layout`。采集事件的重组流程另见[分片重组](shard_reconstruction.md)。

## 如何读结果

转换写出 `checkpoint.json` 与 `.npy`。先确认 `status=complete`，再查看 `parameters`、`unsupported`、`selected_key/selected_prefix` 和源格式；complete 表示所选普通张量成功转换，不表示优化器、RNG 或训练恢复成功。转换 complete 退出 0，incomplete 或处理失败退出 2。

`compare-checkpoint` 输出 `comparison.json/csv`、`report.html`，按参数名匹配，不按文件顺序。可用 `--atol/--rtol`；CLI 没有 `--rules`、`--mapping` 或 `--max-ulp`。浮点规则与[基础比较](compare.md)一致，整数精确、任何非有限值失败。转换遗留的 unsupported 不会被忽略。

比较 pass/fail/incomplete 分别退出 0/1/2。缺参数是 incomplete，shape/dtype/数值不同或损坏数据是 fail。报告里的首个不同参数是按名称组织的观测结果，不是训练执行顺序或根因。

[返回使用手册](../README.md)
