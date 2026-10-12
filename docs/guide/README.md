# musaProbe 功能手册

本手册按功能独立成篇。第一次使用建议先阅读[安装](installation.md)和[快速开始](quickstart.md)，再选择与你的问题有关的功能。示例默认从仓库根目录执行，`/data/musaprobe` 可替换为自己的数据目录。

## 从问题选择入口

| 你想做什么 | 建议阅读顺序 |
|---|---|
| 比较同一模型的两次运行 | [模块采集](features/capture.md) → [运行比较](features/compare.md) → [查看报告](features/reports.md) |
| 定位训练梯度差异 | [反向会话](features/backward.md) → [参数梯度](features/parameter_gradients.md) → [优化器状态](features/optimizer_state.md) |
| 长时间观察数值变化 | [筛选与预算](features/capture_filters.md) → [周期监测与趋势](features/trends.md) → [异常停跑](features/nonfinite_stop.md) |
| 缩小到单个算子 | [API 采集](features/api_capture.md) → [重放](features/replay.md) → [前反向预检](features/precheck.md) |
| 比较不同实现或并行布局 | [训练配置](features/training_config.md) → [名称映射](features/name_mapping.md) → [分片重组](features/shard_reconstruction.md) |

## 采集与控制

| 功能 | 本页讲解 |
|---|---|
| [模块与手动张量采集](features/capture.md) | 最小模型接入、保存中间值和生命周期 |
| [筛选与数据预算](features/capture_filters.md) | step、rank、microbatch、模块筛选与截断 |
| [API 采集](features/api_capture.md) | 高层 API 与 ATen 调用关系 |
| [反向会话](features/backward.md) | 明确反向身份、同图多次 backward 和重计算 |
| [参数梯度](features/parameter_gradients.md) | 更新前梯度、main_grad、分片归属 |
| [优化器状态](features/optimizer_state.md) | 参数及 AdamW 等优化器张量状态 |
| [张量切片](features/tensor_slicing.md) | 按维度保存选定区域及原始形状 |
| [配置热更新](features/hot_reload.md) | 步边界更新、配置版本及多 rank 协调 |
| [基准输入替换](features/input_replacement.md) | 显式干预、结构校验与梯度语义 |
| [非有限值触发与停跑](features/nonfinite_stop.md) | NaN/Inf、有界历史与安全停跑 |
| [异步落盘](features/async_storage.md) | 后台写盘、MUSA D2H、背压及退出排空 |

## 比较与分析

| 功能 | 本页讲解 |
|---|---|
| [运行比较](features/compare.md) | 身份对齐、误差指标与比较结论 |
| [检查与报告](features/reports.md) | inspect、JSON/CSV、HTML 报告 |
| [阈值规则](features/threshold_rules.md) | 模块/API 规则、命令行优先级和 FP32 ULP |
| [显式名称映射](features/name_mapping.md) | 一对一模块/API 映射与拒绝条件 |
| [模糊匹配候选](features/fuzzy_matching.md) | 调用偏移候选与歧义处理 |
| [训练配置检查](features/training_config.md) | 版本、超参、并行、输入和 RNG 证据 |
| [checkpoint 转换与比较](features/checkpoint.md) | 权重键、形状、精度和支持的分片格式 |
| [跨布局分片重组](features/shard_reconstruction.md) | 连续分片与显式矩形布局 |
| [周期监测与趋势](features/trends.md) | 激活、梯度、优化器状态与离线趋势 |
| [执行图与异常传播](features/execution_graph.md) | 可见数据依赖、通信身份和异常候选 |

## 算子与运行模式

| 功能 | 本页讲解 |
|---|---|
| [算子重放](features/replay.md) | 使用实际采集的输入重放支持的纯算子 |
| [前反向精度预检](features/precheck.md) | 构造输入、原采集输入与合成梯度种子 |
| [原生 FP8](features/fp8.md) | 原始编码、scale、amax 和数值解释 |
| [compile 比较](features/compile.md) | eager/compile 输出边界及图内观察 |
| [MUSA Graph 探针](features/musa_graph.md) | 固定形状缓冲区与 replay 新鲜度 |
| [自有内核调试](features/kernel_debug.md) | MUSA/Triton 内核显式调试输出 |

## 框架接入

每种适配单独列出依赖、模型及运行条件。适配代码存在并不意味着所有版本、分布式模式都可运行。

| 功能 | 本页讲解 |
|---|---|
| [Megatron](features/megatron.md) | 训练循环、microbatch、VPP 和梯度分片 |
| [SGLang](features/sglang.md) | 独占 eager Llama 请求的 worker 采集 |
| [VeRL](features/verl.md) | 本地 HF rollout、权重更新和 PPO 示例 |
| [vLLM](features/vllm.md) | worker 扩展接口及版本适配限制 |
| [其他框架与张量导入](features/tensor_import.md) | 后端信息检查与 NumPy 离线导入 |

## 参考

- [配置字段与数据目录](configuration.md)
- [常见问题与退出码](faq.md)
- [更多开发资料](../README.md)

[返回项目首页](../../README.md)
