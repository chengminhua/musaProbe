# 快速开始

这个示例运行一个 `Linear → ReLU → Linear` 小网络，保存前向、反向和参数梯度，再生成正常比较及受控差异报告。它还检查开启采集前、采集中、清理后的输出和梯度是否一致。

## 1. 准备环境与输出目录

先完成[安装](installation.md)，确保当前 Python 已有 PyTorch。下面命令从仓库根目录执行，`/data/musaprobe` 是示例数据根目录，可统一替换为你有权限的路径。

```bash
mkdir -p /data/musaprobe/demo
python3 examples/quickstart.py --device cpu --output /data/musaprobe/demo
```

需要比较 CPU 与 MUSA 时，在已配置好 torch_musa 的设备环境运行：

```bash
python3 examples/quickstart.py --device musa --output /data/musaprobe/demo
```

每次使用固定的 CPU 输入和初始权重，再复制到目标设备。CPU 示例比较两次 CPU 运行；MUSA 示例比较 CPU 参考和 MUSA 运行。该小网络使用 FP32。

## 2. 查看结果

命令打印 JSON 结果和报告的完整路径。新建目录形如：

```text
/data/musaprobe/demo/demo_<时间戳>/
  reference_cpu/             CPU 参考采集
  candidate/                 正常目标设备采集
  injected/                  首层输出增加 0.1 的采集
  comparison_normal/
    comparison.json
    comparison.csv
    report.html
  comparison_injected/
    comparison.json
    comparison.csv
    report.html
  demo_summary.json
```

正常情况下，摘要中 `normal` 为 `pass`，`injected` 为 `fail`，首处观测差异的模块为 `0`。这里的 `fail` 是示例主动制造差异后的预期结果。示例脚本仅在这些断言成立时成功退出。

用浏览器打开两个 `report.html`，比较正常报告与异常报告中的模块、张量路径和误差指标。报告不依赖网络服务。

## 3. 自己执行检查与比较

将下面变量中的 `demo_实际时间戳` 替换为刚才输出的真实目录名：

```bash
RUN=/data/musaprobe/demo/demo_实际时间戳
musaprobe inspect "$RUN/candidate"
musaprobe compare "$RUN/reference_cpu" "$RUN/candidate" \
  --output "$RUN/manual_compare"
```

`inspect` 展示元数据和覆盖情况；`compare` 读取真实张量进行比较。对注入组运行同样命令时，发现差异会返回退出码 `1`：

```bash
musaprobe compare "$RUN/reference_cpu" "$RUN/injected" \
  --output "$RUN/manual_compare_injected"
```

## 4. 接入自己的模型

接下来阅读[模块采集](features/capture.md)中的完整 Python 示例。对自己的两次运行，保持输入、权重、随机状态和训练配置可比，分别使用独立的 `run_id`。先采集少量模块和 step，再扩大观察范围。

结果中出现截断、缺失或未对齐时，先处理证据问题，详见[比较说明](features/compare.md)和[常见问题](faq.md)。这个示例只验证小网络流程，不代表任意模型、精度或并行配置已经通过。

[返回功能手册](README.md)
