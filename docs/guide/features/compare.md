# 比较两次运行的真实张量

`musaprobe compare` 根据事件身份配对两次采集，再逐元素比较保存的张量。适合检查代码修改、设备切换或同一训练配置重复运行后，哪些已采集位置出现变化。

离线比较只需要 NumPy，不需要 PyTorch 或 GPU。输入必须是 musaProbe 采集目录：一个运行根目录中有 `rank_00000/` 等子目录，或直接指定某个 rank 目录；普通训练日志不能作为输入。两边应保存相同范围的数据，报告目录必须位于输入目录之外。

## 先准备一份可运行的数据

已有数据时，直接跳到下一节。下面只用 NumPy 创建三个很小的**合成离线样例**，不是模型运行或设备精度证据。目录已存在时会拒绝覆盖；再次运行请换一个根目录，并同步修改后续命令。

```bash
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/data/musaprobe/manual_demo')
root.mkdir(parents=True, exist_ok=False)
for name, values, module in [
    ('reference_001', [1., 2.], 'encoder.proj'),
    ('candidate_001', [1., 2.1], 'encoder.proj'),
    ('renamed_001', [1., 2.], 'decoder.proj'),
]:
    rank = root / name / 'rank_00000'
    rank.mkdir(parents=True)
    array = np.asarray(values, dtype=np.float32)
    np.save(rank / 'value.npy', array, allow_pickle=False)
    event = dict(schema_version=2, config_version=0, backward_id=None,
                 event_id=0, step=0, microbatch=0, phase='forward',
                 module=module, call_index=0, kind='output', tensor_path='$',
                 op=None, shape=[2], dtype='torch.float32', state='tensor',
                 file='value.npy', device='synthetic_numpy',
                 stats=dict(numel=2, min=float(array.min()), max=float(array.max()),
                            mean=float(array.mean()), l2=float(np.linalg.norm(array)),
                            nan_count=0, posinf_count=0, neginf_count=0))
    (rank / 'events.jsonl').write_text(json.dumps(event) + '\n')
    (rank / 'manifest.json').write_text(json.dumps(dict(
        schema_version=2, rank=0, world_size=1, status='complete',
        events_count=1, config={}, metadata={'source': 'synthetic_numpy_example'})))
print(root)
PY
```

也可以从仓库根目录执行主示例生成真实 CPU 采集，需要已安装 PyTorch：

```bash
python3 examples/quickstart.py --device cpu --output /data/musaprobe/demo
```

该示例会创建 `/data/musaprobe/demo/demo_<时间戳>/`。请使用本次实际目录中的 `reference_cpu`、`candidate`、`injected`，不要把 `/data/musaprobe/demo` 本身当作采集目录。正常对照应通过，`injected` 是故意加入差异的样例。

## 使用方法

```bash
# 自比：预期退出 0，overall=pass。
musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/reference_001 \
  --output /data/musaprobe/manual_reports/self

# 有意改变了一个值：预期退出 1，overall=fail；报告仍会生成。
musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/candidate_001 \
  --output /data/musaprobe/manual_reports/changed
```

参考运行在前，候选运行在后。可选参数有 `--atol`、`--rtol`、`--max-ulp`、`--rules` 和 `--mapping`，分别见[阈值规则](threshold_rules.md)与[显式映射](name_mapping.md)。Python 接口返回包含全部记录的字典：

```python
from musaprobe.compare import compare_runs
result = compare_runs(
    '/data/musaprobe/manual_demo/reference_001',
    '/data/musaprobe/manual_demo/candidate_001',
    output='/data/musaprobe/manual_reports/python_comparison',
)
print(result['overall'], result['first_difference'])
```

## 身份如何配对

先按 rank 区分，再核对 step、microbatch、phase、module、call_index、kind、tensor_path、op、recompute、parameter_name、api_call_index，以及 schema、配置版本、后向 session 和 virtual pipeline 身份。`event_id`、文件名、设备名、运行名不用于跨运行配对。

schema 1 的历史数据仍可读取；schema 2 要求 `config_version`、`backward_id` 等关键字段。两种 schema 不直接混配。`backward_id=null` 不证明多次 backward 已稳定归属；缺省的 `virtual_pipeline` 与 `null` 等价，显式 `0` 与它们不同。重复身份为 `unaligned`，不会静默选取最后一条。

保存切片、梯度范围/阶段/坐标、FP8 编码布局也必须兼容。相同一维 shape 不能证明是同一梯度分片。`world_size` 或已记录的 `metadata.parallelism` 不一致时，不继续逐 rank 数值比较；需要先做[显式分片重组](shard_reconstruction.md)。

## 如何读结果

浮点条件是 `abs(candidate-reference) <= atol + rtol*abs(reference)`；整数、bool 始终精确比较。任何 NaN/Inf 都是异常，即使两边相同。空张量可以通过，但会明确记录比较元素数为 0。

| 总体结果 | 退出码 | 含义 |
|---|---:|---|
| `pass` | 0 | 有比较记录，全部通过，相关清单正常结束 |
| `fail` | 1 | 数值/结构不同，或文件、格式、身份对齐错误 |
| `incomplete` | 2 | 缺项、截断、不支持、只有统计量、空运行或运行未完整结束 |

参数或文件处理错误也可能返回 2，应结合标准错误判断。`fail` 优先于 `incomplete`。`first_difference` 只选数值、NaN/Inf、shape 或 dtype 差异，不把损坏文件和身份错误伪装成首处精度差异；它按观测顺序给出排查入口，不是根因证明。

`summary.verified_tensors` 包含实际检查但失败的张量，不能当作通过数；通过数看 `summary.statuses.pass`。未采集的模块、step、rank、融合核内部不在结论中。完整状态解释和报告入口见[报告与检查](reports.md)。

## 常见问题

- `missing_reference/missing_candidate`：先核对采集范围与调用结构；不要靠改 `event_id` 配对。
- `statistics_only/truncated`：重新采集所需真实张量，或调整经过评估的预算；统计量相同不能证明逐元素相等。
- `unaligned`：检查重复身份、并行布局、schema、配置版本、梯度范围及阶段；名称变化才考虑显式映射。
- NPY shape/dtype 不符或路径越界：修复数据来源。加载使用 `allow_pickle=False`，不会忽略损坏数据继续通过。

[返回使用手册](../README.md)
