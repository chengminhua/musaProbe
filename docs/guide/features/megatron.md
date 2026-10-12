# Megatron 训练采集接入

## 作用与适用场景

Megatron 适配在训练入口与 pipeline schedule 的实际调用点添加观察，记录 step、microbatch、前反向边界，以及优化器即将消费的参数梯度。适合在已有 Megatron + MUSA patch 训练工程中收集可比较的短程证据。

仓库附带的准备和启动脚本针对特定单机 8 卡 Llama3 启动结构；它不是任意 Megatron 仓库的通用训练启动器，也不负责分配集群资源。

## 使用前提与具体支持

训练环境已经能独立运行相应模型，包含匹配的 PyTorch、torch_musa、Megatron 与 MUSA patch。当前源码要求 `train_step` 暴露 model/optimizer、schedule 的 `forward_step` 提供 `current_microbatch`，并能够从真实前向输出对象关联反向；不能识别时拒绝猜测标签。

准备脚本要求源项目具备：

```text
TRAINING_SOURCE/
  code/Megatron-LM/
  code/megatron-lm-musa-patch/
  scripts/run_pretrain_llama3_full.sh
  tokenizer/
  data/deepseekv2_full_epoch/
  hostfile
```

末层启动脚本还必须包含适配器识别的字面量赋值和命令片段。对应示例配置为 32 层、hidden4096、FFN14336、seq4096、TP1/PP2/DP4、MBS1/GBS512、BF16、TE FP8、94720 个训练样本。准备脚本只改隔离副本中的路径、通信入口、观察入口和退出间隔，不自动核验或改写所有模型超参数。

运行前确认 8 张卡及相应 CPU、内存、PID、磁盘资源确实分配给本任务。启动脚本检查显存是否为空和本项目锁，但这不等于资源调度系统授权。

## 准备隔离副本

下面是一套**在已配置训练环境内部执行**的连贯命令，不执行 SSH 或创建容器。先把本项目源码放到 `PACKAGE`，将源训练工程放到 `SOURCE`。三个目录彼此分离；原件与 base 不能互相包含。

```bash
BASE=/data/musaprobe/session
PACKAGE=/data/musaprobe/package
SOURCE=/data/musaprobe/training-source
MASTER_ADDR=localhost
MASTER_PORT=29500

python3 "$PACKAGE/scripts/remote_prepare_llama3.py" \
  --source "$SOURCE" --base "$BASE" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"

cat "$BASE/audit/launch_changes.patch"
```

`localhost` 是单机 rendezvous 示例，可换成当前执行环境可绑定的地址；准备和启动必须完全一致。该步骤完整复制代码、脚本、数据、tokenizer 与 hostfile，核对前后 SHA256；已有 `BASE/workspace` 时拒绝覆盖。复制所需空间取决于实际源项目。

## 预览、短程启动与检查

沿用上一段变量。先预览命令，确认数据、模型和输出配置：

```bash
bash "$PACKAGE/scripts/remote_launch_llama3.sh" \
  --base "$BASE" --package "$PACKAGE" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --run-id statistics_demo --profile statistics --exit-interval 2 --steps 0
```

上面的命令不会启动 torchrun。确认资源和 dry-run 后，显式加 `--launch` 才执行：

```bash
bash "$PACKAGE/scripts/remote_launch_llama3.sh" \
  --base "$BASE" --package "$PACKAGE" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --run-id statistics_demo --profile statistics --exit-interval 2 --steps 0 --launch

python3 "$PACKAGE/scripts/validate_remote_run.py" \
  --base "$BASE" --run-id statistics_demo --require-capture --verify-source
```

`--launch` 还要求至少 500 GiB 可用空间、恰好 8 张卡且检查时显存使用为零、通信端口可绑定。它拒绝重复 run ID。每次实验使用新的 run ID；同一 base 内本项目训练锁防止重复启动。

## 参数与默认采集范围

| 参数 | 含义 |
| --- | --- |
| `--profile baseline` | 不采集，仍使用同一入口和正常退出收尾。 |
| `--profile statistics` / `tensor` | 分别保存统计或实际张量，读取对应 `configs/llama3_*.json`。 |
| `--exit-interval` | 仅接受 `2` 或 `8`。 |
| `--steps` | 零起始、逗号分隔且不重复，必须小于退出间隔；如 `0` 或 `2,3`。 |
| `--expected-hostname` | 可选，校验当前执行环境的 hostname。 |
| `--injection FILE` | 可选受控故障测试，文件必须在 base 内，baseline 不允许；默认关闭。 |

默认捕获每个 PP stage 的 DP0/TP0；候选 rank `[0,4]` 必须通过真实并行坐标核验。默认微批次为 0，模块为本地首层及其 attention、MLP、attention linear_proj，ATen 为 `add.Tensor` 和 `mul.Tensor`。每张量预算 64 MiB，每 rank 8 GiB；BF16 按实际 FP32 保存大小计预算。

多 model chunk 使用 `chunks.N` 模块前缀和 `virtual_pipeline` 身份，必须能从 schedule 的真实 model 参数识别；对象重复、顺序变化或无法关联时拒绝采集。默认脚本不自动启用 VPP。涉及新版 `vp_stage` 的模型构造需由训练入口完整传递，不能事后补属性冒充正确层号。其他 schedule、并行布局与模型仍需独立验证。

## 参数梯度怎样解释

分布式优化器只记录当前 rank 实际拥有、即将消费的梯度分片。适配器递归处理 chained optimizers，读取 `model_param_gbuf_map`、`_get_model_param_range_map(parameter)["param"]` 等私有接口，从 `main_grad` 取 owned 范围，不增加 all-gather。

- `gradient_scope=optimizer_owned_shard`：配合原参数 shape、左闭右开的 `parameter_flat_range` 和并行坐标解读。
- `gradient_stage=post_reduce_scatter_pre_unscale_pre_clip`：尚未执行 unscale、clip 和 optimizer 更新。
- 本 rank 不拥有该参数时写 `unsupported/not_owned_by_rank`；缺接口、范围或 owner 异常不能回退成完整缓冲区。
- 普通优化器记录 `full_parameter`、`pre_unscale_pre_clip`。

累积梯度的 `microbatch=null`，不属于某个单独微批次。完整 `main_grad` 不能直接视为完整全局梯度；其他版本分布式优化器或 FSDP 不在这个私有接口适配的自动支持范围。

## 输出与常见问题

`audit/` 保存原件/副本哈希及 patch；`runs/RUN_ID/` 保存日志、状态、退出码、配置和产品源码哈希；`captures/RUN_ID/rank_XXXXX/` 保存标准采集与 `integration.jsonl`。

`validate_remote_run.py` 的 `inspection_passed` 综合退出状态、真实错误日志、采集错误、阶段覆盖和原件校验。训练 exit 0 不能覆盖日志中的运行时错误；统计摘要一致也不代表实际张量一致。

需要汇总多次运行的训练日志、采集覆盖与耗时时，使用只读汇总脚本，重复 `--run-id` 指定每次真实运行：

```bash
python3 "$PACKAGE/scripts/summarize_validation.py" \
  --base "$BASE" --run-id statistics_demo \
  --output /data/musaprobe/summaries/session
```

输出目录必须在输入 `runs/` 和 `captures/` 之外，生成 `summary.json` 与 `summary.md`。只有存在满足条件的基线与采集运行时才计算耗时对照；汇总不替代张量比较或完整性能测量。

- **准备脚本找不到指定片段**：源启动结构不匹配，应在独立工程中审阅适配，不能删掉匹配检查硬跑。
- **step 与日志 iteration 差 1**：采集取零起始 `args.curr_iteration`，日志一般为 step + 1。
- **没有某个参数的完整梯度**：先看 owned 范围；不要把分片补零后当成完整梯度。
- **预算截断**：`truncated` 表示证据不完整，不能记为数值比较通过。
- **进程退出报错**：正常路径统一同步、barrier、销毁通信组；异常路径不新建 barrier。检查完整日志和清单，不只看训练迭代是否完成。

[返回使用手册](../README.md)
