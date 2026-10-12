# VeRL 本地 HF rollout 与 PPO 采集

## 作用与适用场景

VeRL 接入提供一个由调用方独立持有的 Hugging Face rollout 生命周期适配器，以及真正执行 rollout、log probability 和 PPO 更新的小模型示例。适合检查本地生成与 actor 更新链路、模块前反向采集，以及观察代码是否影响一次 PPO 更新。

当前范围是单进程、未分片的小型 HF 模型；不启动 Ray、HTTP 服务或多机 RL，不支持 FSDP/DDP 包装和服务器 rollout 注册。

## 使用前提与版本契约

需要兼容的 PyTorch、VeRL、Transformers、tokenizers 和相应设备后端。实际 GPU 示例接受 MUSA 或 CUDA；CPU 生命周期测试不能代替 GPU rollout/PPO 执行。

适配所依据的 VeRL `0.8.0.dev0` 接口包括新 `BaseRollout(config, model_config, device_mesh)`、`RolloutConfig`、`HFModelConfig` 和旧 HF `generate_sequences`。该组合使用默认 `mode="async"`，但本工具仍直接调用本地方法，不启动异步服务调度器。`mode="sync"` 已不属于该配置契约。

这一路径依赖 Transformers 的 `AutoModelForVision2Seq`；可用依赖组合包括 Transformers 4.57.0、tokenizers 0.22.2、huggingface-hub 0.36.2。不能假定 Transformers 5 或任意新版 VeRL 直接兼容。请在任务隔离环境中准备依赖；musaProbe 不安装或替换系统包，也不通过伪造类名绕过导入失败。

## 完整小模型示例

以下命令在源码仓库根目录运行。先生成本地随机 tiny Llama 和对应 tokenizer，不下载权重；每次运行使用新的目录。

```bash
PYTHONPATH=src python3 examples/advanced_probe.py make-tiny-model \
  --output /data/musaprobe/verl_tiny_llama

PYTHONPATH=src python3 examples/verl_tiny_step.py \
  --model /data/musaprobe/verl_tiny_llama --device musa \
  --output /data/musaprobe/verl_ppo
```

脚本 `--model`、`--output` 必填，`--device` 为 `musa` 或 `cuda`，默认 MUSA。模型必须为本地 Llama、参数不超过 200 万，且词表、隐藏维、层数和 FFN 大小满足脚本中的小模型限制。

示例执行 2 条 prompt、每条最多 4 个响应 token，使用真实 `HFRollout.generate_sequences`、`DataParallelPPOActor.compute_log_prob` 和 `update_policy`。奖励是由生成 token 构造的合成正数，用于触发 PPO 更新，不评价模型质量。为满足 actor 的 rank 查询，仅创建一个进程的私有 Gloo group。

## 可复用的 rollout 接口

以下是**已有 HF actor/model 上下文片段**。假定 `actor_model` 已在 `device` 上，`model_dir` 指向同一个本地模型配置；rollout 必须独立持有参数存储，不能直接传 actor 自身。

```python
import asyncio
import copy
from verl.workers.config import HFModelConfig, RolloutConfig
from musaprobe.integrations.verl import create_local_hf_rollout

device = "musa"
owned_rollout_model = copy.deepcopy(actor_model)
rollout = create_local_hf_rollout(
    owned_rollout_model,
    RolloutConfig(name="hf", do_sample=False, temperature=1.0,
                  response_length=4, top_p=1.0, top_k=0),
    HFModelConfig(path=model_dir, load_tokenizer=False,
                  trust_remote_code=False,
                  enable_gradient_checkpointing=False,
                  use_remove_padding=False),
    device=device,
    max_state_bytes=64 * 1024**2,
)
try:
    asyncio.run(rollout.release())
    asyncio.run(rollout.resume(["weights", "kv_cache"]))
    asyncio.run(rollout.update_weights(iter(actor_model.state_dict().items())))
    # prompts 必须是符合 HF rollout 契约的真实 VeRL DataProto。
    response = rollout.generate_sequences(prompts)
finally:
    asyncio.run(rollout.release())
```

完整的 DataProto 构造、attention mask、position IDs 和 PPO 输入组装见前面的可执行示例；不能把普通 dict 当作任意版本的 DataProto。

| 接口/配置 | 行为 |
| --- | --- |
| `release()` | 先同步自身后端，再把独立 rollout 模型移到 CPU，校验权重未改变。 |
| `resume(["weights", "kv_cache"])` | 恢复权重到目标设备；动态 KV 在各次生成时建立，不恢复持久静态 cache。 |
| `update_weights(iterable)` | 完整 `(name, tensor)` 迭代器，严格核验名字、shape、dtype、加载结果；失败时回滚自身模型。 |
| `max_state_bytes` | 默认 64 MiB，上限 256 MiB；输入快照和回滚副本分别受预算限制。 |
| `micro_batch_size` | 在对应旧 HF 代码中可缺省并使用请求 batch；显式 `None` 或非正整数会被拒绝。 |

生命周期不能与正在执行的生成并发。模型状态必须为普通稠密未量化 Tensor，拒绝持久 `_cache`、静态 cache 和分布式包装。

## 输出和透明性检查

`verl_step.json` 记录执行阶段、实际后端、参数变化、PPO metrics、rollout 生命周期及采集验证。`captures/verl_step/rank_00000/` 保存 actor 首层 `mlp.down_proj` 的前向和反向边界；actor 会清理参数梯度，因此不能把这些事件称为完整参数梯度采集。

示例在安装 hook 前复制独立的 baseline model/optimizer，逐张量克隆同一批实际 PPO 输入，并核对类型、shape、dtype、device 和存储独立性。采集更新后关闭 probe，恢复更新前 RNG，执行一次无采集对照更新：

- 全部参数必须有限且逐元素精确相等。
- Python、NumPy、PyTorch CPU 和所有可见设备的默认 RNG 更新后状态必须精确相等。
- 采集必须包含完整有限的前向和非零反向文件，反向关联真实前向调用。

证据保存在 `ppo_transparency/`；查看 `parameter_transparency_verified`、`rng_transparency_verified`、`uncaptured_PPO_baseline_compared`。差异保留为 `mismatch`，不会自动放宽阈值。对照后恢复采集侧 RNG，再刷新 rollout 权重并重新生成。

该检查仅覆盖一次本地 PPO 更新和默认生成器，不证明框架私有 RNG、多进程调度、整轮 RL 或任意模型的可复现性。

## 常见问题

- **只有 `framework --framework verl` 通过**：那个入口只验证 CPU DataProto 数据协议，明确不代表 RL 更新成功。
- **HFRollout 抽象方法或构造器不兼容**：使用这里的 `create_local_hf_rollout`，并核对实际安装版本契约；不能删除抽象检查后宣称支持。
- **深拷贝 DataProto 后设备变化**：VeRL 自定义序列化可能改变设备；示例按真实 TensorDict 构造器逐张量克隆，不通过跨设备比较放行。
- **RNG/参数 exact 检查失败**：保留双方原始数组和状态；可能涉及后端确定性，也可能是观察影响，不能直接归因或放宽为通过。

[返回使用手册](../README.md)
