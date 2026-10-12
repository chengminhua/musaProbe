# vLLM worker 扩展接口

## 作用与当前状态

vLLM 适配使用 `worker_extension_cls` 和 `collective_rpc`，让真实 worker 在 `get_model()` 返回的模型上临时安装模块采集，再在请求结束后移除 hook。适合未来在独占 eager 引擎上观察单个请求的浮点模块边界。

**当前属于接口实现，仍待兼容环境中的真实请求采集验证。** 不能因有 Python 入口、CPU 回归或框架发行版可发现，就宣称 vLLM 产品链路已经可用。以下内容用于明确接入契约和验证方式。

## 使用前提与版本边界

需要 vLLM 暴露 `LLM.collective_rpc`、`worker_extension_cls`、worker `get_model()`，且 driver/worker 均能导入同一 musaProbe。已核查的接口来自 vLLM 0.10.1.2 开发版；本示例清理使用 V1 `llm_engine.engine_core.shutdown()`，不承诺其他版本或 V0 具有相同生命周期。

MUSA 平台插件必须与 vLLM、PyTorch、tokenizer 相互兼容，并由真实 `current_platform.device_type` 识别为 MUSA。插件的 attention 构造器、metadata builder 或 cache block 约束不匹配时，应先修复独立的框架环境，不能修改设备名称或伪造硬件架构使验证通过。

Engine 必须由当前任务独占，并使用 `enforce_eager=True`；采集期间不能提交其他请求。张量留在 worker 文件系统，RPC 只传控制信息。

## 独立验证命令

从源码仓库根目录运行。若没有小型本地模型，先在匹配的 Transformers/tokenizer 环境生成；它只创建随机权重。

```bash
PYTHONPATH=src python3 examples/advanced_probe.py make-tiny-model \
  --output /data/musaprobe/vllm_tiny_llama

PYTHONPATH=src python3 examples/vllm_capture.py \
  --model /data/musaprobe/vllm_tiny_llama --device musa \
  --output /data/musaprobe/vllm_capture
```

`--model`、`--output` 必填，设备为 `musa` 或 `cuda`，默认 MUSA。输出目录必须尚未使用。当前命令可能因版本契约不匹配而返回 `unsupported`；保留 `traceback.txt` 和 `vllm_capture.json` 判断具体失败位置。

示例创建自己的单卡 eager Engine，关闭 prefix caching，比较采集前、采集中、移除 hook 后同一贪心请求的 token ID，并要求 worker 有完整采集报告。最大上下文 64 token、最多生成 4 token；这些小规模边界不能代表服务吞吐或所有模型。

## Python 接入契约

以下是**已有本地模型上下文片段**。调用方还需按所用 vLLM 版本管理自己的 Engine 关闭逻辑；可直接参考上述完整示例。

```python
from vllm import LLM, SamplingParams
from musaprobe.integrations.vllm import WORKER_EXTENSION, generate_with_probe

llm = LLM(
    model=model_dir, tokenizer=model_dir, trust_remote_code=False,
    enforce_eager=True, tensor_parallel_size=1,
    max_model_len=64, max_num_seqs=1, max_num_batched_tokens=64,
    gpu_memory_utilization=0.1, enable_prefix_caching=False,
    worker_extension_cls=WORKER_EXTENSION,
)
result = generate_with_probe(
    llm, {"prompt_token_ids": [1, 4, 5]},
    SamplingParams(temperature=0, max_tokens=4),
    config={"output_dir": "/data/musaprobe/vllm_session",
            "run_id": "request_001", "mode": "tensor",
            "modules": ["model.layers.0.mlp.down_proj"]},
    request_tag="request_001", timeout=60,
)
```

`generate_with_probe` 负责控制采集，不关闭调用方的 LLM。请求标记不自动等于框架原生 request ID。每次使用新 run ID，避免与已存在的 worker 输出冲突。

## 参数和返回值

| 项目 | 含义 |
| --- | --- |
| `WORKER_EXTENSION` | worker mixin 的完整导入路径；构造 LLM 时显式传入。 |
| `prompt` | 一个字符串或一个 vLLM prompt dict；不接受多请求列表。 |
| `config.modules` | 模块选择必须匹配真实 worker 模型，未匹配会失败。 |
| `request_tag` | 非空用户标记，最多 256 字符。 |
| `timeout` | worker RPC 超时，须大于 0 且不超过 600 秒，默认 60；不是整个生成的强制超时。 |
| 默认预算 | 每 Tensor 64 MiB、每 rank 256 MiB、10000 事件，均不得超过这些上限。 |

适配只支持 module hooks；拒绝 `apis`、`high_level_apis`、`dependency_graph` 和 `capture_fp8`。RPC 与模型执行可能不在同一线程，不能将线程局部 dispatch 模式直接跨 RPC 使用。

返回值包含 `outputs`、`workers`、`request_tag` 和 `cleanup_verified`。worker 报告记录 rank 输出目录、事件数与关闭状态；标准事件包含 `engine_forward_call`，manifest 在请求结束后补充可获得的原生 request IDs。

## 输出解读与常见问题

`engine_phase` 固定保留 `unknown`，不靠 token 数猜测 prefill/decode。错误请求会尝试清理 hook；RPC 中断导致清理无法确认时，抛出 `CaptureRPCError`，其 `.report` 保留 `cleanup_verified=false`，不能继续声称引擎已恢复干净。

- **平台是 UnspecifiedPlatform 或其他设备**：检查插件注册及导入错误，不只检查是否装了 vLLM。
- **cache block、attention 签名错误**：属于 vendor/core 版本对接问题，适配不会自动修改全局安装或注入兼容补丁。
- **没有采集事件却生成成功**：单请求推理通过不等于 worker 采集通过；检查模块名、worker 扩展和报告状态。
- **相同 token 就能证明完全透明吗**：只能证明该请求的输出 token 一致，不覆盖所有 logits、随机状态或并发服务行为。

[返回使用手册](../README.md)
