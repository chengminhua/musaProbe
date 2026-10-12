# 按改动选择验证

命令从仓库根目录执行，使用已准备好依赖的 Python。不要默认安装、升级 PyTorch 或设备运行时。测试文件对应关系见[代码导航](repository-map.md)。

## 文档和 agent 资料

```bash
python3 skills/musaprobe-development/scripts/check_docs.py
```

此脚本仅使用标准库，读取待发布 Markdown，检查相对链接/锚点、Python/JSON 代码块以及可用时的 `bash -n`，并提示个人目录与内网地址。它不会执行代码块、连接服务器或写文件。自动检查不是凭据和业务内容的完整审计，也不验证外部链接或示例的运行语义。

完整 CPU/NumPy 示例在可写的独立目录运行，按需替换示例中的 `/data/musaprobe`；依赖已有模型的片段先检查上下文和参数，不伪装成独立可执行程序。文档改动不要求启动 GPU。

## 相关回归

本项目使用标准库 `unittest`，按实际改动选择文件。例如改对齐判定：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_compare.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_analysis_compare.py' -v
```

涉及公共采集/存储、身份格式或发布时，相关检查通过后再跑完整回归：

```bash
MUSAPROBE_TEST_TE_RUNTIME=0 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

多数测试是 CPU/fixture；完整套件需要 PyTorch，部分可选依赖缺失会 skip。`test_native_fp8.py` 的真实 MT-TE 导入由 `MUSAPROBE_TEST_TE_RUNTIME=1` 显式启用，可能初始化设备运行时，不能在普通 CPU 验证中随意开启。记录实际测试数量、失败和 skip 原因，不硬编码过去的通过数。

## 端到端小模型

```bash
PYTHONPATH=src python3 examples/quickstart.py \
  --device cpu --output /data/musaprobe/dev-check/quickstart
```

目录自动增加时间戳；预期正常比较 `pass`、注入组 `fail` 且首处差异为首层，同时确认输出/梯度透明性。不要把主动注入组的 `fail` 当作测试自身失败。

其他针对性入口：

- 采集生命周期：[capture_extensions_check.py](../../../examples/capture_extensions_check.py)，显式 `--device cpu`。
- 激活/梯度/AdamW 周期采集：[periodic_monitor_check.py](../../../examples/periodic_monitor_check.py)，显式 `--device cpu`；趋势保留缺口可以是预期 `incomplete`。
- 编译边界：[advanced_probe.py](../../../examples/advanced_probe.py) 的 `compile --device cpu --backend aot_eager`，只证明 CPU 路径。
- 图内探针：[graph_internal_check.py](../../../examples/graph_internal_check.py) 的 `--mode compile --device cpu`，默认设备是 MUSA，必须显式覆盖。

上述脚本还需要 `--output`；先看帮助，使用新的输出目录。纯设备示例、框架服务和多进程通信实验按[设备实验约定](device-lab.md)单独安排。

Megatron 适配改动还可运行独立的 [CPU 契约验证](../../../scripts/validate_megatron_integration.py)，检查 microbatch 关联、梯度归属、异常收尾和独立入口；其数据在临时目录生成，不启动真实训练：

```bash
PYTHONPATH=src python3 scripts/validate_megatron_integration.py
```

## 依赖与安装包

修改离线模块时，在只有 NumPy 的环境中验证相应 CLI，并确认正常导入没有加载设备运行时：

```bash
PYTHONPATH=src python3 - <<'PY'
import sys
import musaprobe
import musaprobe.compare
import musaprobe.report
assert 'torch' not in sys.modules
assert 'torch_musa' not in sys.modules
print(musaprobe.__version__)
PY
```

这条快速检查只覆盖所导入模块，不能替代 NumPy-only 环境中所修改命令的实际执行。

构建前确认 setuptools 77+、wheel 和 packaging 24.2+ 已安装；构建依赖以 `pyproject.toml` 为准。依赖不足时使用独立构建环境，不升级系统或设备环境：

```bash
python3 -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /data/musaprobe/build/wheels .
```

发布相关修改还应把产物安装到独立环境，离开源码目录检查 CLI。CLI 成功、CPU 实验、真实 MUSA、实际多 rank 和框架请求是不同验证层级，报告中保留区别。复杂修改可使用[验收记录模板](../assets/validation-record.md)，记录原始证据而非只写“测试通过”。
