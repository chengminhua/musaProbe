# 生成模糊匹配候选

`suggest-mapping` 为未按原身份配对的事件列出待人工审查的候选。适合模块改名较多、不容易逐个寻找对应位置时使用。它比较元数据，不读取张量做数值验收，也不会自动接受任何配对。

需要两份 musaProbe 采集，离线只依赖 NumPy。可运行[基础比较](compare.md)的小样例，比较原名与改名数据。

## 使用方法

```bash
musaprobe suggest-mapping /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/renamed_001 \
  --output /data/musaprobe/manual_reports/mapping_candidates.json
```

`--output` 是可选的 JSON 文件；不指定时只输出到终端。该文件用于审查，不能直接作为 `compare --mapping` 的输入。

Python API 可以设置候选与搜索预算，CLI 没有对应选项：

```python
from musaprobe.analysis import suggest_mapping
result = suggest_mapping(
    '/data/musaprobe/manual_demo/reference_001',
    '/data/musaprobe/manual_demo/renamed_001',
    max_candidates=3, max_pairs=10000,
)
print(result['accepted'], result['suggestions'])
```

`max_candidates` 每条最多显示多少候选，范围 1～100，默认 5；`max_pairs` 是本次全局最多评估的候选对数，默认 100000，必须为正整数。

## 候选依据

先限定同 rank、兼容并行布局、相同执行上下文、shape、逻辑 dtype、算子、切片与梯度布局，再依据模块名相似度和调用序号距离排序。原本已经唯一配对的事件会跳过。上下文或布局不兼容不会因为名字接近而得到建议。

`score` 是排序分数，不是正确概率，也不是精度指标。数值完全不同的两份张量也可能具有高分。只有一个候选也不能自动认定语义相同。

## 如何读结果

顶层和每条建议均保留 `accepted=false`，顶层 `requires_explicit_mapping=true`。

| 建议状态 | 下一步 |
|---|---|
| `candidate` | 只有一个已评估候选，仍需人工核对 |
| `ambiguous` | 多个候选，结合实际代码、上下文和调用结构确认 |
| `unmatched` | 当前兼容范围内没有候选，检查采集范围或结构变化 |
| `budget_exhausted` | 搜索预算耗尽，候选列表可能不完整 |

查看 `candidates[*].evidence`、`candidate_count`、`examined_pairs`、`truncated`、`rejected_ranks` 和 `coverage`。`candidate_count` 是实际评估到的数量；预算耗尽时不能据此断言没有其他对应关系。

生成候选正常返回 0，即使包含 unmatched、ambiguous 或预算耗尽；这不是精度 pass。参数或文件处理错误为 2。确认关系后，手工写成[显式映射](name_mapping.md)的 `pairs` 格式，再执行真实张量比较。

## 常见错误

- 自比没有建议：通常是原身份已唯一配对，并非工具没运行。
- 名字很相似但没有候选：检查 phase、op、shape、配置版本、后向身份和并行布局。
- 把候选 JSON 直接交给 `--mapping`：两种文件结构不同，必须先审查并填写明确规则。
- 提高预算后想直接宣布通过：预算只扩大搜索范围，仍未执行数值比较。

[返回使用手册](../README.md)
