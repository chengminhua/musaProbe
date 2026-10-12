# 比较训练配置与环境证据

`check-config` 检查两份记录中的版本、训练参数、并行配置、精度、输入摘要和随机状态是否一致。适合在解释数值差异之前先排除明显的实验条件变化。

它比较已经记录的字段，不会登录机器补查环境，也不会恢复随机状态。离线比较只依赖标准库/NumPy，可输入采集目录、JSON 对象文件，或在 Python API 中直接传字典。

## 使用方法

先生成[基础比较](compare.md)的合成采集，再执行：

```bash
musaprobe check-config /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/candidate_001 \
  --output /data/musaprobe/manual_reports/config.json
```

这份合成样例没有记录训练环境，因此应为 unknown、退出 2，而非 equal。检查真实训练时，把两个输入替换为已采集的实际路径；下面介绍如何补充配置证据。

没有采集时，可用以下最小例子直接观察不同参数和缺失证据如何分别处理：

```python
from musaprobe.analysis import compare_configs

reference = {'training': {'lr': 0.001}, 'rng': None}
candidate = {'training': {'lr': 0.002}, 'rng': None}
result = compare_configs(reference, candidate, fields=['training.lr', 'rng'])
print(result['overall'])  # different
for row in result['records']:
    print(row['field'], row['status'], row['unknown_paths'])
# training.lr different []
# rng unknown ['rng']
print(compare_configs(reference, reference, fields=['training.lr'])['overall'])  # equal
print(compare_configs(reference, reference, fields=['rng'])['overall'])  # unknown
```

API 的 `fields` 是非空、无重复的点分字段路径列表。CLI 没有 `--fields`，使用默认字段；如果要自定义检查范围，使用上述 Python API。默认字段位于 manifest 的 `metadata` 下：

```text
python_version, torch_version, torch_musa_version, driver_version,
transformer_engine_version, parallelism, training, precision,
input_state, random_state
```

直接输入 JSON 文件时也按相同点分路径查找，因此用于默认 CLI 检查的文件应有 `metadata` 外层。多 rank 采集逐 rank 对应检查；缺失某个 rank 的字段显示 unknown。

## 在采集前明确记录配置

采集侧可调用辅助函数，需要 PyTorch。下面的 `input_tensor` 是程序已有输入，`probe_config` 是程序已有的采集配置；把这段放在创建调试器之前，并按实际训练填写参数：

```python
from musaprobe import PrecisionDebugger
from musaprobe.environment import capture_training_config

metadata = capture_training_config(
    hyperparameters={'learning_rate': 0.001},
    parallelism={'tensor': 1, 'pipeline': 1, 'data': 1},
    precision={'dtype': 'float32'},
    inputs={'sample': input_tensor},
    include_rng=True,
    max_input_bytes=64 * 1024**2,
)
probe = PrecisionDebugger(config=probe_config, metadata=metadata)
```

训练参数、并行设置和精度信息由调用方传入，不会自动遍历训练对象。输入仅对显式传入的 dense Tensor 做 shape/dtype/SHA256 摘要，默认总预算 64 MiB；复制到 CPU 可能同步设备，应在有意选择的安全边界调用。

RNG 记录 Python、NumPy、PyTorch CPU 状态摘要，以及已初始化加速后端的**当前设备**状态。不会为检查而初始化 GPU，不覆盖全部自建生成器或所有设备，也不是可恢复的 RNG checkpoint。缺少包、驱动字段或设备状态时，后续显示 unknown 是有效结果。

## 如何读结果

| 状态 | CLI 退出码 | 含义 |
|---|---:|---|
| `equal` | 0 | 指定字段都有证据且相等 |
| `different` | 1 | 至少存在明确差异 |
| `unknown` | 2 | 存在缺失/未知信息，且没有已知差异 |

参数或文件处理失败也返回 2。每条记录保留 `reference/candidate`、`reference_present/candidate_present`、`different_paths` 和 `unknown_paths`。同一对象既有已知差异又有未知子项时，状态为 different，但未知路径仍保留。

缺字段、`null`、空对象和显式 `status=unknown` 都不算相等，即使两侧都缺失。递归检查值的类型，整数 `1` 与布尔 `true` 不相等。

## 支持范围与常见错误

equal 只证明本次请求检查的配置证据一致，不证明输入之外的全部数据、模型权重、整个环境或数值结果一致。权重另用[checkpoint 比较](checkpoint.md)，张量另用[基础比较](compare.md)。

若两次相同采集仍是 unknown，先查看缺失路径；例如未初始化 CUDA 的 `random_state.cuda=null` 会保留未知。确实只关心某个已记录设备时，可通过 API 明确选择 `metadata.random_state.musa.sha256`，不要把未采集字段伪填为相同值。输出文件应在输入采集目录之外。

[返回使用手册](../README.md)
