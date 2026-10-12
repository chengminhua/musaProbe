# 安装

musaProbe 的离线分析与设备采集分开加载依赖。只查看或比较已有数据时，无需安装 PyTorch 或 MUSA 运行时。

## 环境要求

| 使用场景 | 依赖 |
|---|---|
| 比较、报告、趋势等离线分析 | Python 3.10+、NumPy 1.23+ |
| PyTorch/Megatron checkpoint 转换 | 上述依赖，以及支持源格式的 PyTorch；格式要求见 [checkpoint 指南](features/checkpoint.md) |
| PyTorch CPU/CUDA 模型采集与算子执行 | 上述依赖，以及能运行目标模型的 PyTorch 环境 |
| MUSA 模型采集与算子执行 | 上述依赖，以及互相匹配的 PyTorch、torch_musa、驱动和运行库 |
| 源码打包 | setuptools 77+、wheel、packaging 24.2+ |
| FP8、compile、训练或推理框架适配 | 对应功能页指定的 Transformer Engine、编译器或框架依赖 |

musaProbe 不负责安装或升级设备驱动、PyTorch、torch_musa。设备侧请沿用已能正常运行模型的环境；用同一个 Python 安装和执行工具。

## 从源码安装

将仓库下载到自己的工作目录，在仓库根目录执行：

```bash
python3 -m pip install .
python3 -m musaprobe --version
musaprobe --help
```

项目展示名与仓库名为 `musaProbe`，安装包、Python 导入和命令名使用小写 `musaprobe`。例如 Python 中使用 `from musaprobe import PrecisionDebugger`。

普通安装会解析项目声明的 NumPy 依赖。若当前环境已具有所需依赖与构建工具，希望直接使用现有环境：

```bash
python3 -m pip install --no-build-isolation --no-deps .
```

`--no-deps` 要求你自行保证依赖完整，`--no-build-isolation` 要求当前环境已有 setuptools 77+、wheel 和 packaging 24.2+；其中 packaging 用于校验许可证表达式。缺少时使用普通安装的隔离构建环境，或在自己的虚拟环境中补齐构建依赖。

## 离线部署

在具备构建依赖的环境打包：

```bash
python3 -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /data/musaprobe/build/wheels .
```

将生成的 wheel 复制到目标环境。确认目标 Python 已具备 NumPy，再安装指定版本文件，例如：

```bash
python3 -m pip install --no-index --no-deps \
  /data/musaprobe/build/wheels/musaprobe-0.1.0-py3-none-any.whl
python3 -m musaprobe --version
```

设备及框架依赖应在目标环境单独准备，不能通过复制 musaProbe 的 wheel 获得。

## 确认安装

离线功能检查：

```bash
python3 -m musaprobe --help
```

CPU 采集检查（需 PyTorch，在源码仓库根目录执行）：

```bash
python3 examples/quickstart.py --device cpu --output /data/musaprobe/demo
```

MUSA 环境可先检查框架自身能否使用设备：

```bash
python3 -c 'import torch, torch_musa; print(torch.__version__); print(torch.musa.is_available())'
```

返回 `True` 说明框架可见 MUSA；具体算子与框架兼容性仍取决于目标环境。接下来运行[快速开始](quickstart.md)。

## 开发安装

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

设备和可选框架测试有各自前提。分析机缺少对应依赖时，不能把跳过测试解读成设备能力通过。

[返回功能手册](README.md)
