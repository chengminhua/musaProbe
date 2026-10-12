# 按明确布局重组分片

分片重组把多个 rank 保存的局部数组恢复成声明的完整张量，便于后续数值比较。适合已知优化器拥有的连续 DP 范围，或已明确每片全局偏移的矩形布局。

离线只需要 NumPy。必须有完整张量文件、正常结束的清单、可验证的布局和一致执行上下文。工具检查物理覆盖，不猜测参数名、转置规则或 TP/PP/VPP 的模型语义。输出要求新目录或空目录，并位于输入及布局文件之外。

## 两种入口

| 命令 | 输入布局依据 |
|---|---|
| `reconstruct` | 事件中的 `optimizer_owned_shard`、原始形状、连续 flat range 和 TP/PP/DP 坐标 |
| `reconstruct-layout` | 用户声明 `global_shape` 与每个 `(rank,event_id)` 的逐轴 offsets |

第一种只合并固定 TP/PP 等坐标内的 DP 分片；不同 TP/PP 组输出为不同合成 rank。第二种允许来源模块名和 virtual pipeline 不同，但必须由你明确给出全局名字和偏移。

## 可复制的小例子

下面生成两个合成分片，分别包含 `[0,1,2]`、`[3,4,5]`，每片实际 shape 为 `[1,3]`。它仅演示布局合同，不是分布式训练证据。

```bash
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/data/musaprobe/manual_shards')
root.mkdir(parents=True, exist_ok=False)
for rank_id in (0, 1):
    rank = root / f'rank_{rank_id:05d}'
    rank.mkdir()
    array = np.arange(rank_id * 3, rank_id * 3 + 3, dtype=np.float32).reshape(1, 3)
    np.save(rank / 'grad.npy', array, allow_pickle=False)
    event = dict(schema_version=2, config_version=0, backward_id=None,
                 event_id=0, step=0, microbatch=None, phase='optimizer',
                 module='projection', parameter_name='projection.weight',
                 call_index=0, kind='parameter_grad', tensor_path='$', op=None,
                 shape=[1, 3], dtype='torch.float32', state='tensor', file='grad.npy',
                 stats={'numel': 3}, gradient_scope='optimizer_owned_shard',
                 original_parameter_shape=[2, 3],
                 parameter_flat_range=[rank_id * 3, rank_id * 3 + 3],
                 gradient_stage='post_reduce_scatter_pre_unscale_pre_clip',
                 gradient_parallel_coordinates={'tensor': 0, 'pipeline': 0, 'data': rank_id})
    (rank / 'events.jsonl').write_text(json.dumps(event) + '\n')
    (rank / 'manifest.json').write_text(json.dumps(dict(
        schema_version=2, rank=rank_id, world_size=2, status='complete',
        events_count=1, config={}, metadata={'source': 'synthetic_shard_example'})))
PY

musaprobe reconstruct /data/musaprobe/manual_shards \
  --parameter projection.weight --step 0 --max-bytes 1048576 \
  --output /data/musaprobe/manual_reports/reconstructed_flat
```

`--parameter` 是精确参数名，`--step` 是非负采集步编号；两者省略时处理所有符合条件的组。`--max-bytes` 默认 1 GiB，限制重组输出总数组大小。

同一份样例也能按二维矩形明确重组：

```bash
mkdir -p /data/musaprobe/manual_rules
cat > /data/musaprobe/manual_rules/axes.json <<'JSON'
{
  "schema_version": 1,
  "groups": [{
    "name": "projection.weight", "global_shape": [2, 3],
    "shards": [
      {"rank": 0, "event_id": 0, "offsets": [0, 0]},
      {"rank": 1, "event_id": 0, "offsets": [1, 0]}
    ]
  }]
}
JSON

musaprobe reconstruct-layout /data/musaprobe/manual_shards \
  --layout /data/musaprobe/manual_rules/axes.json --max-bytes 1048576 \
  --output /data/musaprobe/manual_reports/reconstructed_axes
```

`offsets` 是每一轴在完整数组中的起点，片的尺寸取自实际文件，不另填长度。这里 rank1 的 `[1,0]` 表示第二行起点，不是展平下标 3。同一事件不能反复用于多个位置或多个组。

## 如何读结果

两个入口分别保存 `reconstruction.json`、`axis_reconstruction.json`，以及可供 `inspect/compare` 读取的 rank 目录和 NPY。先检查顶层 `status`、每组 `records[*].status/reason`、`sources` 和 `errors`。样例应 complete，重组值为 `[[0,1,2],[3,4,5]]`。

complete 退出 0，incomplete 或参数处理错误退出 2。某些组成功不代表所有组成功；缺片不会补零。输出保留来源 rank、事件和范围/偏移，方便追溯。

两次运行应分别按同一明确合同重组，再用 `compare` 比较对应输出。不同入口生成的合成身份和并行元数据可能不同，不保证可直接互比。重组 complete 只证明布局和文件覆盖，不能保证数值正确；NaN/Inf 或两次运行的数值差异仍需比较检查。

## 支持范围与常见错误

连续模式要求 flat range 是原始参数展平后的 `[start,end)`，长度等于保存元素数，各片共同完整覆盖；`not_owned_by_rank` 不补成零。重复 ownership、阶段不一致、未知坐标或缺片都不能完成。

逐轴模式要求矩形在全局范围内、互不重叠、体积完整覆盖，dtype 和 step、phase、microbatch、配置版本、后向 session、schema、梯度/优化器阶段一致。允许显式归组并不证明你的参数命名判断正确。

两种模式目前均拒绝已经切片的载荷和编码 FP8 分片；预算不足也保持 incomplete。修改原始形状或偏移来“消除缺口”会改变声明含义，应回到采集和分片来源核实。

[返回使用手册](../README.md)
