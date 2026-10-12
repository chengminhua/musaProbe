# 容差、FP32 ULP 与模块/API 规则

阈值决定已正确配对的张量何时被判为不同。适合为特定模块或 API 指定经过评估的误差上限，或用 FP32 ULP 补充绝对/相对误差检查；不能用阈值修复身份、shape 或布局不一致。

需要两份可比较的采集，离线执行只需 NumPy。[基础比较](compare.md)提供 `reference_001` 和 `candidate_001` 合成样例，其模块名为 `encoder.proj`。

## 基本用法

```bash
musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/candidate_001 \
  --atol 0 --rtol 0 --max-ulp 0 \
  --output /data/musaprobe/manual_reports/strict
```

上面的有意差异应退出 1。逐元素浮点条件为：

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

| 逻辑 dtype | 默认 atol | 默认 rtol |
|---|---:|---:|
| FP16 | `1e-3` | `1e-3` |
| BF16 | `1e-2` | `1e-2` |
| FP32、FP64 | `1e-5` | `1e-4` |

`atol/rtol` 必须有限且非负；整数和 bool 始终精确比较，不受宽松浮点阈值影响。BF16 即使以 FP32 文件保存，仍按逻辑 BF16 处理。任意 NaN/Inf 都不会通过。

FP32 会计算 ULP，`--max-ulp` 必须为非负整数，并增加一个**同时需要满足**的上限。它不会替代 `atol/rtol`；正负零 ULP 为零。给非 FP32 张量应用 ULP 上限会得到 unsupported，FP8 解码后的值也不能冒充原生 FP32 精度。混合 dtype 采集宜用下面的 dtype 规则限制 ULP 适用范围。

## 按模块或 API 配置

```bash
mkdir -p /data/musaprobe/manual_rules
cat > /data/musaprobe/manual_rules/thresholds.json <<'JSON'
{
  "schema_version": 1,
  "rules": [
    {"id": "projection", "module": "encoder.proj", "atol": 0.2, "rtol": 0},
    {"id": "matmul_fp32", "op": "aten.mm.*", "dtype": "torch.float32", "max_ulp": 16}
  ]
}
JSON

musaprobe compare /data/musaprobe/manual_demo/reference_001 \
  /data/musaprobe/manual_demo/candidate_001 \
  --rules /data/musaprobe/manual_rules/thresholds.json \
  --output /data/musaprobe/manual_reports/rules
```

这个教学规则会让约 0.1 的合成差异通过，只用于验证规则生效，不是推荐生产阈值。第二条只在采集中存在相应 FP32 API 事件时命中。

每条规则至少一个选择器 `module/op/dtype`、至少一个阈值 `atol/rtol/max_ulp`。选择器是大小写敏感的 glob，多个选择器按 AND 匹配；没有匹配记录的规则不会凭空增加比较。`id` 应唯一，省略时自动生成。

优先级为：显式 CLI 值 > 命中规则 > dtype 默认值。可以只覆盖一个值，例如追加 `--atol 0`，其他阈值保留规则结果。两个命中规则给同一阈值设置不同值会报错，即使 CLI 另有覆盖也先拒绝冲突；不会按规则顺序取最后一个值。

## 如何读结果

在 `comparison.json.records[*]` 查看 `threshold_policy.matched_rules` 和 `sources`，确认规则命中及显式覆盖。`metrics` 保留实际 `atol`、`rtol`、`max_ulp_threshold`、`max_ulp_error`、`ulp_mismatch_count` 和 `tolerance_mismatch_count`。ULP 与容差异常可能落在相同元素，不能把两种计数简单相加当作总异常数。

规则按照配对后的参考事件字段选择；使用名称映射时，要核对参考端名字。结果退出码沿用 compare：pass 0、fail 1、incomplete 2；规则格式或冲突错误为 2。

## 常见错误

- 规则中的 `dtype` 写成 `float32`，实际事件是 `torch.float32`：glob 匹配的是事件原字段，应照事件填写。
- 大范围 `max_ulp` 命中 FP16/BF16：增加 `dtype: "torch.float32"` 选择器。
- 两条规则重叠且阈值冲突：拆分选择范围或统一阈值，不依赖顺序。
- 为了得到 pass 不断放宽阈值：先查输入、配置、配对和数据范围，保留差异证据再决定容差。

[返回使用手册](../README.md)
