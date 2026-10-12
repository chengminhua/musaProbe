---
name: musaprobe-development
description: 在 musaProbe 仓库实现或审查采集、精度分析、框架适配和文档改动，选择对应验证并遵守数据身份与设备实验边界。用于本仓库开发维护，不替代普通用户功能手册。
---

# musaProbe 开发

这是随仓库维护的开发上下文。相对路径从当前文件解析，命令从仓库根目录执行；它依赖本仓库源码和手册，不是可以脱离仓库工作的独立工具包。

展示名与仓库名使用 `musaProbe`，Python 包、导入和 CLI 使用小写 `musaprobe`。软件版本与数据 schema 独立，调整发布版本不得重置数据格式或修改已有身份字段。

## 从任务选择资料

| 当前工作 | 按需阅读 |
|---|---|
| 找到实现、修改某项能力 | [代码与测试导航](references/repository-map.md) |
| 改采集、存储、身份对齐或数值判定 | [关键行为约定](references/invariants.md) |
| 选择回归、检查文档或制作安装包 | [验证方法](references/validation.md) |
| 操作 MUSA、多 rank、训练/推理服务或远程实验 | [设备实验约定](references/device-lab.md) |
| 给复杂改动留下可追溯证据 | 按需复制 [验收记录模板](assets/validation-record.md) 到任务输出目录 |

## 工作方法

1. 核对任务范围、当前分支和已有修改，找到有关实现、测试与功能手册。版本和能力从当前文件确认，不沿用历史测试计数或机器空闲结论。
2. 针对可观察的行为实现改动。新增接口同时考虑正常输入、受控错误和失败路径；设备或框架能力需要真实执行证据，不能仅用 mock、CPU 或占位接口代替。
3. 保持离线分析依赖边界、采集透明性、可靠身份和严格失败判定。细节按修改领域查阅，不重新发明文件格式或状态含义。
4. 先运行相关 CPU/离线验证。涉及运行时或多卡时，在已有任务授权下按设备约定逐级扩大；遇到客观阻塞记录具体原因，继续独立条目。
5. 接口、默认值、CLI 或支持范围改变时，同步 `docs/guide/` 对应功能页；新增功能页加入索引。README 保持产品入口，避免变成实验流水账。
6. 交付说明实际改变、验证结果和剩余限制。跳过、未知、部分通过与完整验收区分；保存必要命令、退出码和证据路径。

## 立即可用的入口

- 产品使用：[功能手册](../../docs/guide/README.md)、[配置说明](../../docs/guide/configuration.md)。
- 代码导航：[src/musaprobe](../../src/musaprobe)、[tests](../../tests)、[examples](../../examples)。
- 只检查文档，不执行文档里的示例：

```bash
python3 skills/musaprobe-development/scripts/check_docs.py
```

不要把真实主机、凭据或临时实验结果补进 skill。无需为了读取此目录安装全局配置；不支持 skill 发现的 agent 也可从根目录 `AGENTS.md` 顺着链接使用。
