# 代码与测试导航

产品代码在 `src/musaprobe/`，`tests/` 是 unittest 回归，`examples/` 是可运行的功能示例与验证；当前公开用法在 [docs/guide](../../../docs/guide/README.md)。源码树不保存调研下载、历史实验输出或构建缓存。

下表源文件相对 `src/musaprobe/`，测试模块相对 `tests/`。按本次改动选相关项，不把表中所有运行时都作为默认测试要求。

| 修改领域 | 源文件 | 测试模块 |
|---|---|---|
| 配置/筛选 | `config.py`、`capture_extensions.py` | `test_config`、`test_capture_extensions` |
| 模块/API/反向生命周期 | `capture.py`、`capture_extensions.py` | `test_capture`、`test_capture_extensions` |
| 落盘/预算/异步快照 | `storage.py`、`async_transfer.py` | `test_storage`、`test_async_d2h`、`test_capture_extensions` |
| 对齐/指标/阈值/名称映射 | `compare.py`、`policies.py` | `test_compare`、`test_analysis_compare` |
| CLI/报告 | `cli.py`、`report.py` | `test_cli`、`test_report` |
| 配置证据/趋势/匹配候选/依赖图 | `analysis.py`、`graph.py`、`environment.py` | `test_offline_analysis`、`test_environment` |
| checkpoint/分片重组 | `checkpoint.py`、`distributed_checkpoint.py`、`layout.py`、`layout_axes.py` | `test_checkpoint_layout`、`test_distributed_checkpoint`、`test_layout_axes` |
| 重放/预检 | `replay.py`、`precheck.py` | `test_replay`、`test_precheck` |
| FP8 编码/NumPy 解码 | `native_fp8.py`、`fp8_numpy.py` | `test_native_fp8`、`test_fp8_numpy`、`test_fp8_capture_acceptance` |
| compile/图内探针 | `compiled.py`、`graph_probe.py` | `test_execution`、`test_graph_probe` |
| 多 rank 安全边界/通信观察 | `distributed.py`、`integrations/communication.py` | `test_distributed`、`test_integration_contracts` |
| Megatron/VPP/优化器归属 | `integrations/megatron.py`、`integrations/megatron_entry.py`、`integrations/optimizer.py` | `test_integration_contracts`、`test_optimizer`、`test_training_state_example`、`test_checkpoint_lifecycle_example` |
| SGLang/vLLM/VeRL | `integrations/` 下同名适配及 `sglang_models/` | `test_sglang_integration`、`test_vllm_integration`、`test_verl_integration`、`test_verl_capture_check` |
| 其他框架/内核入口 | `frameworks.py`、`kernel_debug.py` | `test_frameworks`、`test_kernel_interfaces` |
| 周期采集示例 | 仓库根的 `examples/periodic_monitor_check.py` | `test_periodic_monitor_example` |
| 隔离复制/训练启动参数 | 仓库根的 `scripts/remote_prepare_llama3.py`、`remote_launch_llama3.sh` | `test_remote_parameters` |

一次运行多个相关测试可用：

```bash
PYTHONPATH=src:tests python3 -m unittest test_config test_capture_extensions -v
```

`tests/` 不是 Python 包，不要假定 `tests.test_capture` 总能导入。也可使用 [验证方法](validation.md)中的 `unittest discover -p`。

## 测试名不等于设备验收

- `test_async_d2h` 使用 CPU transport fixture，不证明 MUSA DMA 已执行。
- `test_execution` 包含 CPU `aot_eager`，不证明 MUSA Inductor。
- 框架适配测试大量使用契约 fixture，不证明真实 worker 请求或完整训练可运行。
- checkpoint 与 FP8 的部分测试有可选 PyTorch 分支；NumPy-only 环境中的 skip 要单独报告。
- `test_remote_parameters` 创建临时假工程并 dry-run，不启动真实训练。

## 文档更新位置

新用户可见功能写入 `docs/guide/features/<功能>.md`，在 `docs/guide/README.md` 加入口；先解释作用和前提，再给例子、参数、输出与限制。配置字段同步 `docs/guide/configuration.md`，安装依赖变化同步 `docs/guide/installation.md`。改动影响开发流程或这些文件导航时，再更新本 skill，避免复制整份用户手册。
