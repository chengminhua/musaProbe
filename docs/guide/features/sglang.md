# SGLang 单请求 worker 采集

## 作用与适用场景

SGLang 适配通过公开的 external model package 入口，在真实 Llama worker 中临时安装模块 hook。它适合观察一个由当前任务独占的 eager 请求，并将各次模型前向与原生 `EXTEND`、`DECODE` 模式关联。

这一入口只支持独占、单 worker、单请求 Llama 场景。它不是通用在线服务采集，也不使用 token 数量猜测 prefill/decode 阶段。

## 使用前提

需要已兼容的 SGLang/MUSA/PyTorch 环境、本地 `LlamaForCausalLM` 模型及 tokenizer，driver 和 worker 都能导入同一 musaProbe 包并访问输出目录。

当前适配依据 SGLang `0.5.6.post2` 的 external model package、Llama model 与 `ForwardBatch` 接口；不声明任意后续版本兼容。更换版本时，需要核对这些接口以及 Engine 参数、模型模块名称。

必须先创建 `SGLangCaptureSession`，再导入 SGLang model registry、构造 Engine。进程不能已有另一个 `SGLANG_EXTERNAL_MODEL_PACKAGE` 或采集 session；适配不修改已安装的 vendor 源码。

以下条件由适配器逐项核验：TP/PP/DP/节点数均为 1，最多一个运行请求，关闭 CUDA/MUSA Graph、overlap schedule、radix cache、torch.compile、speculative decoding 和 disaggregation；模型不带 quantization 配置。

## 完整验证入口

所有命令在源码仓库根目录执行。如果没有可用的 tiny 模型，可先在匹配框架依赖的环境中生成随机权重，无需下载：

```bash
PYTHONPATH=src python3 examples/advanced_probe.py make-tiny-model \
  --output /data/musaprobe/tiny_llama

PYTHONPATH=src python3 examples/sglang_capture.py \
  --model /data/musaprobe/tiny_llama \
  --output /data/musaprobe/sglang_capture
```

示例固定 MUSA，要求模型配置明确为 Llama，hidden 不超过 1024、层数不超过 4。它创建和关闭自己的 Engine，分别运行采集前、采集中、采集后的相同贪心请求，并检查 token ID 一致、采集窗口之外没有新增文件，以及真实 worker 与 driver PID 不同。

已有 tiny 模型可以复用，不必再次生成；每次采集必须更换新的输出目录，session 不覆盖旧目录。

## 接入自己的独占请求

以下是**已有本地模型上下文片段**；`model_dir` 是兼容的本地 Llama/tokenizer 路径。必须在新的独立进程中执行，不能给正在服务其他请求的引擎临时套上该 session。

```python
from musaprobe.integrations.sglang import SGLangCaptureSession

session = SGLangCaptureSession(
    "/data/musaprobe/sglang_session",
    config={"modules": ["model.layers.0.mlp.down_proj"]},
    max_forward_calls=8,
)
engine = None
try:
    from sglang import Engine
    engine = Engine(
        model_path=model_dir, tokenizer_path=model_dir, device="musa",
        trust_remote_code=False, tp_size=1, pp_size=1, dp_size=1,
        context_length=64, mem_fraction_static=0.1,
        max_total_tokens=128, max_running_requests=1,
        max_prefill_tokens=64, chunked_prefill_size=64,
        disable_cuda_graph=True, disable_overlap_schedule=True,
        disable_radix_cache=True,
    )
    result = session.generate(
        engine, input_ids=[1, 2, 3],
        sampling_params={"temperature": 0, "max_new_tokens": 4},
        request_tag="request_001",
    )
    print(result["cleanup_verified"])
finally:
    try:
        if engine is not None:
            engine.shutdown()
    finally:
        session.close()
```

`close()` 恢复 session 设置的环境变量并关闭控制窗口，不负责替使用者关闭 Engine，所以关闭顺序如上。

## 参数与预算

| 参数 | 含义 |
| --- | --- |
| `output_dir` | 新建 session 目录，worker 必须可读写。 |
| `config.modules` | 明确的 tensor-only 叶模块选择；默认首层 `mlp.down_proj`，不允许 `<root>`。 |
| `config.mode` | 仅 `tensor`。 |
| `max_tensor_bytes` | 默认及上限均为 1 MiB。 |
| `max_rank_bytes` | 默认及上限为每次前向 4 MiB。 |
| `max_events` | 默认及上限为每次前向 128。 |
| `async_write` | 可选后台磁盘写入；不开放其他 dispatch/FP8 配置。 |
| `max_forward_calls` | `1..16`，默认 `8`；请求 `max_new_tokens` 必须不超过它。 |
| `request_tag` | 用户提供的非空标记，最多 256 字符。 |

每个 session 只允许一次采集请求。一个请求的多次模型 forward 会生成多个独立 capture；预算不是整台服务的全局预算。

## 输出解读

示例根目录有 `framework.json`、`output_ids.npy` 和 `session/`。session 保存 `bootstrap.json`、`control.json`、`worker_ready.json`、`request.json`、`forwards/forward_XXXX.json` 及 `captures/forward_XXXX/rank_00000/`。

事件记录真实 `native_forward_mode`、worker PID、forward index 和 `request_tag`。请求关联来自调用方承诺的独占窗口；不声称在模型边界获得了原生请求 ID 路由。`cleanup_verified` 要求 worker 报告确认 hook 移除；状态异常会抛出 `SGLangCaptureError`，其 `report` 保留可用证据。

## 常见问题与边界

- **registry 已导入**：启动新的独立进程并调整初始化顺序，不修改全局 registry 绕过检查。
- **没有匹配模块**：模块名以真实 worker 模型为准，不能直接套用任意 Hugging Face 模型名称。
- **没有 DECODE 事件**：普通请求可能提前结束；完整验证示例使用 `ignore_eos=True`，要求实际观察到 EXTEND 和 DECODE。适配器不会虚构缺失阶段。
- **想开连续批处理、多 worker 或量化模型**：当前明确不支持，需要额外生命周期、请求关联和布局适配。
- **只有 `framework --framework sglang` 通过**：那只证明一个本地推理请求执行成功，不等于 worker 内采集通过。

[返回使用手册](../README.md)
