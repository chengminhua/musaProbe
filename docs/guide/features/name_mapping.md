# 显式名称与调用身份映射

显式映射告诉比较器：两侧哪些不同的模块名或调用字段代表同一语义位置。适合代码重构后模块改名、已确认等价的参数路径变化；前提是你已经确认计算含义相同，而非仅观察到 shape 相同。

它只影响比较时的有效身份，不修改原始采集。离线只需要 NumPy。可先运行[基础比较](compare.md)的小样例：`reference_001` 中模块为 `encoder.proj`，`renamed_001` 中为 `decoder.proj`，数值相同。

## 使用方法

```bash
mkdir -p /data/musaprobe/manual_rules
cat > /data/musaprobe/manual_rules/mapping.json <<'JSON'
{
  "schema_version": 1,
  "pairs": [
    {
      "id": "projection_rename",
      "reference": {"module": "encoder.proj", "kind": "output"},
      "candidate": {"module": "decoder.proj", "kind": "output"}
    }
  ]
}
JSON

musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/renamed_001 \
  --mapping /data/musaprobe/manual_rules/mapping.json \
  --output /data/musaprobe/manual_reports/mapped
```

这个样例映射后应为 pass。去掉 `--mapping` 会保留两侧缺项，而不会因为数值或 shape 相同自动配对。

## 哪些字段可以写

可改变的字段为 `module`、`op`、`parameter_name`、`tensor_path`、`call_index`、`api_call_index`。选择器是**精确值匹配，不支持 glob**。两侧必须写相同的字段集合。

可以附加 `rank/step/microbatch/phase/kind/recompute/config_version/backward_id/virtual_pipeline` 来缩小范围，但这些上下文字段两侧值必须相同，不能用映射跨 rank、跨 step 或把前向改成反向。

一个模块映射规则可以覆盖多个 step 或调用，由剩余上下文继续逐条唯一配对；它不是把多个事件合并成一个。规则 ID 和两侧选择器不得重复；同一事件被多个规则命中会拒绝。映射后出现重复有效身份也为 `unaligned`。

## 如何读结果

`records[*].identity` 是比较使用的有效身份；`mapping.reference/candidate.original_identity` 保留原身份，`mapping.*.rule` 保留规则 ID，双方原始事件也仍在报告中。

两侧映射必须成对命中。同一规则只有一侧命中、规则完全没有选中某侧事件，都不能作为成功映射。配对后仍然检查 shape、dtype、切片、FP8 格式、梯度范围/阶段及并行布局，任何不兼容都不会被映射掩盖。

CLI 退出码沿用 compare。重复身份或未命中规则产生 `unaligned`，通常使比较退出 1；非法规则、重叠选择器等处理错误退出 2。不能只根据是否生成报告判断映射有效。

## 支持范围与常见错误

这是用户声明的身份关系，不证明两段实现数学等价，不会转置、reshape、拼接、转换 dtype 或重放计算。需要更换分片布局时，先看[分片重组](shard_reconstruction.md)。

名字很多时可用[模糊匹配候选](fuzzy_matching.md)辅助寻找，但候选文件不是可直接传给 `--mapping` 的规则。若报 `selected no events`，先用 `inspect` 或 JSON 查看真实模块名、kind 和调用字段；不要为了消除错误扩大到含义不明的范围。

[返回使用手册](../README.md)
