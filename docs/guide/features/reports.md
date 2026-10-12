# 检查采集并阅读报告

`inspect` 用于先确认采集到了什么，`report` 用于把已有比较 JSON 再生成 HTML。适合检查预算和覆盖、筛选异常记录、分享可离线打开的报告。两者只需要 NumPy 和标准库。

先有采集目录或 `compare` 生成的 `comparison.json`。可使用[基础比较](compare.md)的小样例；本页路径沿用其输入和输出。

## 使用方法

```bash
musaprobe inspect /data/musaprobe/manual_demo/reference_001

musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/candidate_001 \
  --output /data/musaprobe/manual_reports/changed

# 上一条因有意差异退出 1；报告已生成，可单独继续执行本条。
musaprobe report /data/musaprobe/manual_reports/changed/comparison.json \
  --output /data/musaprobe/manual_reports/changed/copied_report.html
```

`inspect` 仅接收采集路径，没有 `--output` 参数；需要保存终端 JSON 时使用 shell 重定向。`report` 必须指定 `--output` 文件名。重建 HTML 不会重新比较 NPY，也不会修改比较 JSON 的结论。

## 生成了哪些文件

| 文件 | 阅读方式 |
|---|---|
| `comparison.json` | 完整机器可读结果：`overall`、`summary`、`coverage`、`records`、`first_difference` |
| `comparison.csv` | 逐记录身份、状态、主要指标，适合表格筛选 |
| `report.html` | 单文件交互页面，可直接打开，无 CDN 或服务器依赖 |

终端 `compare` 输出是摘要，完整逐条记录在 JSON 文件中。CSV 会处理可能触发表格公式的文本；HTML 会转义文本，并将超过 JavaScript 安全整数范围的数显示为字符串，避免大整数被舍入。

HTML 支持按 rank、状态和搜索词筛选，按模块路径展开层级，查看误差指标、双方事件、调用栈、映射及梯度分片元数据。需要浏览器启用 JavaScript 才能使用交互筛选；关闭时仍可查看 JSON/CSV。

## 如何判断报告里的异常

先看 `coverage` 和 manifest 是否完整，再看记录状态及 `reason`，最后看指标。`mismatch` 是超出容差，`nonfinite` 是 NaN/Inf，`shape_mismatch/dtype_mismatch` 是结构不同。`error/unaligned` 也使总体失败，但不等于模型精度有缺陷。

`missing_*`、`statistics_only`、`truncated`、`unsupported` 表示证据不完整。主动筛掉的 rank 不算已验证：两侧均有正常结束的空清单、且配置明确排除该 rank 时，不会仅因这些空清单使整体 incomplete。可看 `configured_rank_filter`、`excluded_by_rank_filter` 和 `recorded_ranks`。

| 指标 | 含义与注意点 |
|---|---|
| `max_abs_error/mean_abs_error` | 成对有限元素的最大/平均绝对误差 |
| `max_relative_error` | 以参考值为分母；参考为零而候选非零时无界 |
| `mismatch_count/mismatch_ratio` | 不满足规则的元素数及比例，包含异常的非有限元素对 |
| `cosine_similarity/relative_l2` | 辅助指标，不独立决定 pass |
| `nonfinite_reference/nonfinite_candidate` | 两侧真实张量非有限元素数 |
| `first_mismatch_flat_index` | 第一个异常元素的 C 顺序展平下标 |

JSON 中 `null` 或 HTML 的 `—` 不表示误差为零。无界相对误差同时给出 `relative_error_unbounded=true`，相对 L2 有对应标记。存在 NaN/Inf 时，部分指标只使用成对有限元素，应优先看异常状态。

## 支持范围与常见错误

`inspect` 读取清单和事件元数据，不逐个打开 NPY 验证内容；退出 0 只说明检查动作完成。`report` 退出 0 只说明 HTML 已生成；参数/路径处理失败为 2。不要用两者的退出 0 代替 `compare` 的精度结论。

模块路径树不是执行依赖图。需要实际张量生产者或显式通信边时，使用[执行依赖图](execution_graph.md)。首处观测差异也不是跨 rank 的因果根节点。

报告可能包含采集机器路径、调用栈和环境信息，分享前检查实际内容。移动 HTML 不需要同步 NPY，但再运行数值比较仍需要原始采集。把日志目录传给 `inspect`、把 HTML 传给 `report` 都不是有效输入。

[返回使用手册](../README.md)
