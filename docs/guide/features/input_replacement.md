# 显式模块输入替换

输入替换把一次指定模块调用的某个输入张量换成给定数据，用于验证“如果该位置的输入正确，后续差异是否消失”。这是主动干预，会改变输出和梯度路径，应单独运行实验，不能用于证明采集透明性。

前提是知道精确模块名、输入叶路径、step、rank 和 microbatch，并且替换数据与原张量形状、dtype、layout 一致。以下例子完整展示一次 CPU 干预，输出路径可配置。

## 最小示例

```python
import json
import torch
from musaprobe import PrecisionDebugger

model = torch.nn.Identity()
x = torch.ones(2, requires_grad=True)
with PrecisionDebugger(config={
    "output_dir": "/data/musaprobe/input_replacement",
    "mode": "tensor", "rank": 0, "world_size": 1,
}) as probe:
    probe.enable_input_replacement([{
        "module": "<root>", "tensor_path": "args[0]",
        "step": 0, "rank": 0, "microbatch": 0,
        "value": torch.full((2,), 4.0),
    }], enabled=True)
    probe.start(model)
    probe.set_context(step=0, microbatch=0, phase="forward")
    changed = model(x)
    changed.sum().backward()
    unchanged = model(x)  # 规则已使用，第二次不再替换。
    assert torch.equal(changed, torch.full((2,), 4.0))
    assert torch.equal(unchanged, x) and torch.equal(x, torch.ones(2))
    assert x.grad is None

manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
print(manifest["metadata"]["input_replacement"])
print(probe.rank_dir)
```

替换生成新的 autograd 叶张量，不修改原输入存储。示例中原始 `x` 没有经过已替换的反向路径，因此 `x.grad is None` 是预期结果。

## 关键接口与规则

`enable_input_replacement(rules, enabled=True)` 必须显式启用。每条规则只能包含下面六个字段；模块和路径使用精确名称，不使用 glob。

| 字段 | 说明 |
|---|---|
| `module` | 模型根为 `<root>`，子模块采用 `named_modules()` 的实际路径。 |
| `tensor_path` | 位置参数如 `args[0]`；字典关键字参数叶如 `kwargs['hidden']`。 |
| `step`、`rank`、`microbatch` | 都必须是非负整数；`microbatch=None` 不能作为替换坐标。 |
| `value` | Tensor，或数字 `.npy` 文件路径。 |

传入 Tensor 时，规则保存一份 detach 后的克隆。`.npy` 使用 `allow_pickle=False` 读取；不能用 pickle 或对象数组作为输入。匹配时再次检查 shape/dtype/layout，再复制到原输入所在设备，保持原输入的 `requires_grad` 设置。

规则只使用一次；相同坐标下重复调用模块时，第一次匹配调用消耗规则。再次调用 `enable_input_replacement` 会替换当前规则列表，不是追加。阶段为 `backward` 时不应用替换，避免把规则误用于已标注的重计算。

替换规则在模块 pre-hook 中独立匹配，普通 `modules`、step 筛选不是干预的启停开关。不要仅靠把模块从采集范围排除来禁用替换；最清楚的方式是给干预实验创建独立采集器。

## 输出如何看

替换前后分别记录 `input_replacement_before` 和 `input_replacement` 事件，携带 `intervention.type="input_replacement"` 与 `gradient_semantics="replacement_is_new_leaf"`。

关闭后，manifest 的 `metadata.input_replacement` 给出配置数量 `configured`、应用数量 `applied` 和未匹配规则 `unmatched`。检查这一项，才能区分“干预确实执行”和“规则根本没有命中”。张量文件仍受[存储预算](capture_filters.md)限制；干预执行本身不保证所有前后张量都能完整落盘。

## 支持范围与常见错误

只支持明确的 Tensor 叶及普通 tuple/list/dict 容器，不自动替换任意自定义对象。一个调用中，多条规则必须各自命中不同叶路径；重复路径、缺失路径或 shape/dtype/layout 不一致会拒绝该次替换，而不是广播、改 dtype 后继续。

共享模块的规范路径和调用次数要先通过[模块采集](capture.md)确认。没有命中时，重点检查 microbatch 是否仍是默认的 `None`、当前 step/rank 是否正确，以及根模块是否写成了空字符串。

[返回使用手册](../README.md)
