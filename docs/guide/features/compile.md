# torch.compile 输出验证与显式图内探针

## 作用与适用场景

编译验证将同一模型分别交给 eager 与 `torch.compile`，使用至少两组不同输入比较输出，并确认图确实进入所选编译后端。适合排查编译前后数值变化，避免只调用了编译 API 却实际上没有图进入后端。

如果掌握模型源码，还可以插入固定形状的 `GraphTensorProbe`，观察选定的图内模型张量。它不自动展开融合 kernel，也不捕获任意闭源算子的内部变量。

## 使用前提

需要提供 `torch.compile` 的 PyTorch。CPU 的 `aot_eager` 可用于接口检查；MUSA 需要对应 torch_musa 与编译工具链。`torchada` 选项还需要已安装、能选择 MUSA 的 TorchAda。各运行应使用独立进程和未使用的输出目录，命令从源码仓库根目录执行。

本页没有名为 `musaprobe compile` 的主 CLI 子命令，运行入口是 `examples/advanced_probe.py` 或 Python API。

## 先运行完整小模型检查

```bash
PYTHONPATH=src python3 examples/advanced_probe.py compile \
  --device cpu --backend aot_eager \
  --output /data/musaprobe/compile_cpu

PYTHONPATH=src python3 examples/advanced_probe.py compile \
  --device musa --backend torchada \
  --output /data/musaprobe/compile_musa
```

示例创建 Linear + GELU 的小模型，使用固定输入及变化输入。CPU 通过不能代替 MUSA 编译器通过。也可以显式指定 `--backend inductor`，但工具不会在它失败时自动换成其他后端。

`torchada` 是本项目的适配选项：先导入 TorchAda，核对其选择 MUSA、已应用补丁，再调用实际 Inductor。报告分别记录 `adapter` 与 `resolved_backend=inductor`；它不是 Dynamo 注册的同名编译器。TorchAda 的导入有进程级影响，因此应在任务自己的进程内使用。

## Python API：验证自己的模型

下面是完整 CPU 示例；将设备和后端改为 MUSA 前，应先确认相应环境可用。

```python
import torch
from musaprobe.compiled import validate_execution

model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU())
inputs = (torch.arange(8, dtype=torch.float32).reshape(2, 4),)
result = validate_execution(
    model, inputs, "/data/musaprobe/compile_custom",
    device="cpu", mode="compile", backend="aot_eager",
    replays=3, atol=1e-5, rtol=1e-4,
)
print(result["status"], result["compiled_graphs"])
```

接口会分别 deepcopy 两个模型副本，切到 eval 并在 `no_grad` 下运行；原模型应能被正常 deepcopy。输入只能是非空 tuple/list 中的普通浮点 Tensor，输入总量不超过 64 MiB，模型必须返回一个 Tensor。

| 参数 | 含义 |
| --- | --- |
| `device` | `cpu`、`musa`、`cuda`。 |
| `backend` | 默认 `inductor`；`torchada` 仅允许 MUSA compile。 |
| `replays` | Python API 接受 `2..16`，默认 `2`；高级示例固定使用默认值。 |
| `atol`、`rtol` | 默认 `1e-5/1e-4`，保存在比较指标中。 |
| `output_dir` | 新目录，保存 JSON 与两侧 NumPy 数组。 |

输入在第 `i` 次执行时增加 `i * 0.125`，因此这一工具检查的是受控变化输入，而非任意用户提供的测试数据集。

## 显式观察模型图内张量

以下是**已有模型上下文片段**：假定 `linear` 的输出固定为 `(2, 3)`、FP32、MUSA。将探针作为 `self` 子模块创建，在模型的 `forward()` 方法内部调用：

```python
from musaprobe.graph_probe import GraphTensorProbe

class ObservedBlock(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.point = GraphTensorProbe((2, 3), dtype=torch.float32, device="musa")

    def forward(self, inputs):
        hidden = self.point(self.linear(inputs))
        return torch.nn.functional.gelu(hidden)

model = ObservedBlock(linear)
```

调用方负责创建并启动 `debugger`、配置实际编译后端，把**同一个 `model`** 编译为 `execute`。随后在编译图之外执行与保存：

```python
output = execute(inputs)
model.point.save(debugger, "linear.internal", execution_mode="compile")
```

不要把 `point.save()`、CPU 拷贝或 JSON 写入放进被编译的 forward。可直接运行完整接入示例：

```bash
PYTHONPATH=src python3 examples/graph_internal_check.py \
  --mode compile --device musa \
  --output /data/musaprobe/compile_internal
```

该示例在 MUSA 上导入 TorchAda 后使用 Inductor；CPU 用 `--device cpu`，实际后端为 `aot_eager`。探针显式复制中间值到固定设备 buffer，并递增 generation；它会增加额外操作和存储，不能视为零开销观察。

## 输出与常见问题

边界验证生成 `execution.json`、`input_I_J.npy`、`eager_I.npy` 和 `candidate_I.npy`。`status=pass` 要求全部输出比较通过且 `compiled_graphs>0`；`unsupported` 保留实际异常原因，不意味着所有版本都不支持。图内示例生成 `graph_probe.json` 和 `capture/rank_00000/`，检查三次不同输入、generation 递增、旧快照读取被拒绝。

- **fullgraph 编译失败**：当前验证刻意使用 `fullgraph=True`；先查看不支持的算子或 Python 行为，不会静默退回 eager。
- **shape 改变后探针报错**：探针固定 shape/dtype/device，需重新创建匹配的实例；不支持一个 buffer 任意追随动态形状。
- **能否验证训练完整图**：当前边界工具是 eval 输出验证，图内示例也不能代表反向、优化器 step 或所有动态 shape 的支持。
- **重复 snapshot 报错**：默认只允许读取新 generation；先完成下一次真实执行，不要把旧数据当作新结果。

[返回使用手册](../README.md)
