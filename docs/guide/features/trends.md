# 查看跨 step 的统计趋势

`trend` 将同一事件身份在多个 step 的统计量连成序列，帮助观察均值、范围、L2 或 NaN/Inf 数量随时间变化。适合周期采集、定位异常从何时开始出现，或检查采样是否留下缺口。

它只使用事件中已经保存的统计量，不重新读取 NPY 计算，也不执行两次运行的精度比较。离线仅依赖 NumPy。需要多个 step 的记录；单 step 数据可以生成页面，但无法形成时间变化证据。

## 从真实 CPU 周期采集开始

已有 PyTorch 时，从仓库根目录运行以下小模型示例；输出目录必须尚不存在：

```bash
python3 examples/periodic_monitor_check.py --device cpu \
  --output /data/musaprobe/periodic_demo
```

它实际执行 6 个训练 step，配置只在 step 1、3 保存模块激活、反向值、参数梯度和 AdamW 张量状态。其余 step 继续训练，且与无采集基线核对输出、参数、梯度、优化器状态和 RNG；这不是长期训练性能测试。

先看 `periodic_demo/summary.json` 的 `status` 和 `assertions`，再打开 `periodic_demo/trends/trends.html`；原始数据位于 `periodic_demo/captures/periodic`。示例检查通过时退出 0，但趋势本身预期为 incomplete：显式声明的 6 个 step 中，未采集位置仍保留为空档，特别是 step 2，不补成零。

接入自己的训练时，周期参数见[筛选与预算](capture_filters.md#周期采集)，AdamW 等状态保存见[优化器状态](optimizer_state.md)。应由训练循环更新 step；趋势命令不会启动训练或自动补采历史数据。

## 一个明确保留缺口的例子

下面只写统计记录，不伪造真实张量文件。step 1、3 有数据，step 2 没采集。

```bash
python3 - <<'PY'
import json
from pathlib import Path

rank = Path('/data/musaprobe/manual_trend/rank_00000')
rank.mkdir(parents=True, exist_ok=False)
events = []
for event_id, step in enumerate((1, 3)):
    events.append(dict(schema_version=2, config_version=0, backward_id=None,
        event_id=event_id, step=step, microbatch=0, phase='forward',
        module='projection', call_index=0, kind='output', tensor_path='$', op=None,
        shape=[2], dtype='torch.float32', state='statistics',
        stats=dict(numel=2, min=float(step), max=float(step), mean=float(step),
                   l2=(2 ** 0.5) * step, nan_count=0, posinf_count=0, neginf_count=0)))
(rank / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
(rank / 'manifest.json').write_text(json.dumps(dict(
    schema_version=2, rank=0, world_size=1, status='complete', events_count=2,
    config={}, metadata={'source': 'synthetic_statistics_example'})))
PY

# 预期退出 2：中间缺一采样点，但仍生成 JSON 和 HTML。
musaprobe trend /data/musaprobe/manual_trend \
  --output /data/musaprobe/manual_reports/trend
```

CLI 必须指定输出目录，生成 `trends.json`、`trends.html`；没有选择单个指标或 step 范围的 CLI 参数。Python 接口是 `build_trends(capture, output=...)`，位于 `musaprobe.analysis`。

## 如何分组、如何确定缺口

同 rank 内按除 step 外的语义身份分组，还核对 shape、dtype、切片和梯度布局。因此不同 microbatch、调用序号、配置版本或显式后向 session 不会混成一条线。某些身份只出现在一个 step 时会得到稀疏序列，不能擅自抹掉这些字段强行连线。

若 manifest 中有非空显式 `config.steps`，期望 step 是该列表与实际观测 step 的并集；否则是实际最小到最大 step 的完整整数区间。所以上例产生 step 2 的 missing。若你明确配置 `steps=[1,3]`，列表之外的 step 2 不会被当作应采缺口。隐式区间跨度超过 100000 时会拒绝，需要明确采样步列表。

## 如何读结果

可选指标为 `min/max/mean/l2/nan_count/posinf_count/neginf_count`。HTML 可选序列和指标、点击点查看详情，每条序列最多展示前 2000 个点，完整数据在 JSON。

| 点状态 | 含义 |
|---|---|
| `available` | 存在可用统计量 |
| `nonfinite` | 已记录 NaN/Inf，属于异常点 |
| `missing` | 期望 step 没有该身份记录 |
| `duplicate` | 同一序列在该 step 有多条记录，未静默选一条 |
| `statistics_missing/statistics_invalid` | 统计量缺失或存在非法统计值 |
| `truncated/unsupported/error` | 保留采集时的不可用状态 |

缺口显示为断线，不补零、不插值。`metrics` 中缺失的单个指标为 null；即使其他指标可用，这一指标也不能画成零。

`status=complete` 表示有序列、清单完整且点状态都为 available/nonfinite，退出 0；存在缺口等则 incomplete，退出 2。**complete 不代表数值健康**：非有限点可以存在于完整趋势中。应同时检查 `summary.nonfinite`、各点状态和相关采集记录。

## 支持范围与常见错误

只有文件没有统计量，不能从此命令得到重新计算的曲线；需要采集时保存统计。状态完整也不证明两次运行逐元素一致，后者用[基础比较](compare.md)。

只有一个点或很多分散序列时，先核对 microbatch、call_index、backward_id 和配置版本。故意稀疏采集时出现 incomplete 可能正是应保留的证据，不应通过填零或删点把它改成 complete。

[返回使用手册](../README.md)
