# 配置与数据目录

`PrecisionDebugger` 接受字典、`ProbeConfig` 对象或 JSON 文件。配置决定保存哪些事件、保存多少数据；不会自动替你调用训练循环中的 step、microbatch 或反向边界。

## 最小配置

以下是已有模型接入片段，`model` 是你已构建的 PyTorch 模型：

```python
from musaprobe import PrecisionDebugger

with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/captures",
    "mode": "tensor",
    "steps": [0],
}) as probe:
    probe.set_context(step=0, microbatch=0, phase="forward")
    probe.start(model)
    # 在此执行 model 的前向；完整示例见“模块采集”。
```

单进程省略 `run_id` 时自动生成唯一目录名。分布式进程必须提供相同的显式 `run_id`，或在启动所有进程前设置共享的 `MUSAPROBE_RUN_ID`。

## 使用 JSON 文件

将以下内容保存到 `probe.json`：

```json
{
  "output_dir": "/data/musaprobe/captures",
  "mode": "statistics",
  "steps": [0, 1],
  "modules": ["*"],
  "max_tensor_bytes": 67108864,
  "max_rank_bytes": 8589934592
}
```

在已有模型代码中替换初始化方式：

```python
from musaprobe import PrecisionDebugger

probe = PrecisionDebugger(config_path="probe.json")
```

`config_path` 与 `config` 不能同时提供。完整基础配置见 [configs/example.json](../../configs/example.json)。修改文件不会自动生效；运行中更新使用[配置热更新](features/hot_reload.md)接口。

## 基础字段

| 字段 | 默认值 | 含义 |
|---|---|---|
| `output_dir` | 必填 | 输出根目录 |
| `run_id` | 自动生成 | 一次采集的目录名；多 rank 共享 |
| `mode` | `statistics` | 统计摘要；`tensor` 额外保存张量载荷 |
| `steps` / `ranks` / `microbatches` | `null` | 不筛选；列表指定非负编号，从 0 开始 |
| `modules` | `["*"]` | 模块名通配符 |
| `exclude_modules` | `[]` | 排除的模块名通配符 |
| `apis` / `high_level_apis` | `[]` | 选定 ATen / 高层 API；默认不启用 |
| `capture_inputs` / `capture_outputs` | `true` | 模块输入/输出 |
| `capture_gradients` | `true` | 张量反向 hook；参数梯度快照仍需显式调用 |
| `capture_parameters` | `false` | 参数快照 |
| `stack_depth` | `8` | 调用栈深度，`0` 关闭 |
| `rank` / `world_size` | `null` | 自动从进程组或环境读取；显式值须与进程组一致 |

## 预算与扩展字段

| 字段 | 默认值 | 说明与详细用法 |
|---|---|---|
| `max_tensor_bytes` | 64 MiB | [单张量载荷预算](features/capture_filters.md)，JSON 中填写字节数 |
| `max_rank_bytes` | 8 GiB | 每 rank 的累计张量载荷预算 |
| `max_events` | `100000` | 普通事件预算；耗尽时可额外产生一条终止通知 |
| `tensor_slices` | `[]` | [切片规则](features/tensor_slicing.md) |
| `monitor_every_n_steps` | `null` | [周期监测](features/trends.md)间隔；`null` 关闭 |
| `monitor_start_step` / `monitor_max_steps` | `0` / `100` | 监测起点和采样步数上限 |
| `history_size` / `stop_on_nonfinite` | `0` / `false` | [有界历史与非有限值停跑](features/nonfinite_stop.md) |
| `async_write` / `async_d2h` | `false` / `false` | [后台写盘与 MUSA 异步搬运](features/async_storage.md) |
| `async_queue_size` | `4` | 有界队列容量 |
| `dependency_graph` | `false` | [可见数据依赖](features/execution_graph.md) |
| `capture_fp8` | `false` | [FP8 编码与缩放状态](features/fp8.md) |

未知配置字段、非法类型和无效切片会被拒绝。`async_d2h=true` 要求同时开启 `async_write`。输入替换通过显式 Python 接口启用，不是默认配置动作。

张量预算按实际保存表示计算。例如 BF16 普通张量以 FP32 数值载荷保存，同时保留逻辑 dtype。预算不是整个进程的内存上限，也不包含任意业务模型、系统缓存或所有元数据开销。

## 数据结构

```text
output_dir/
  run_id/
    rank_00000/
      manifest.json        版本、配置、运行状态和数据量
      events.jsonl         事件身份、统计、张量路径及状态
      tensors/             保存的张量载荷
    rank_00001/
      ...
```

已有 rank 目录不会被覆盖。结束时使用上下文管理器或 `close()` 完成排空和清理，`stop()` 用于暂停 hook；详细生命周期见[模块采集](features/capture.md)。

新采集记录采用 schema 2，分析端继续读取 schema 1。旧数据若缺少可靠对齐所需的身份或布局信息，不能通过补造字段强行配对。原始张量、调用栈和训练配置可能包含业务内容，分享时请先检查副本。

[返回功能手册](README.md)
