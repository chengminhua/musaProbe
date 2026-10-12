"""Command line entry points; analysis commands do not load a GPU runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="musaprobe", description="musaProbe 精度采集与离线分析工具")
    parser.add_argument("--version", action="version", version=f"musaProbe {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="查看采集状态、覆盖与非有限值")
    inspect.add_argument("capture")
    compare = commands.add_parser("compare", help="比较两次运行并生成 JSON/CSV/HTML")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--output", required=True)
    compare.add_argument("--atol", type=float)
    compare.add_argument("--rtol", type=float)
    compare.add_argument("--max-ulp", type=int, help="FP32 ULP 上限")
    compare.add_argument("--rules", help="模块/API 阈值规则 JSON")
    compare.add_argument("--mapping", help="显式一对一身份映射 JSON")
    report = commands.add_parser("report", help="从 comparison.json 重新生成 HTML")
    report.add_argument("comparison")
    report.add_argument("--output", required=True)
    replay = commands.add_parser("replay", help="重放指定纯 ATen 调用；默认 CPU 参考")
    replay.add_argument("rank_dir")
    replay.add_argument("--event-id", type=int, required=True)
    replay.add_argument("--device", choices=("cpu", "musa", "cuda"), default="cpu")
    replay.add_argument("--reference-dtype", choices=("same", "float32"), default="same")
    replay.add_argument("--atol", type=float)
    replay.add_argument("--rtol", type=float)
    replay.add_argument("--output")
    configs = commands.add_parser("check-config", help="比较版本、训练配置与随机状态证据")
    configs.add_argument("reference")
    configs.add_argument("candidate")
    configs.add_argument("--output")
    mapping = commands.add_parser("suggest-mapping", help="生成需人工确认的结构匹配候选")
    mapping.add_argument("reference")
    mapping.add_argument("candidate")
    mapping.add_argument("--output")
    trends = commands.add_parser("trend", help="从周期采集记录生成离线趋势")
    trends.add_argument("capture")
    trends.add_argument("--output", required=True)
    graph = commands.add_parser("graph", help="分析实际数据依赖与非有限值传播候选")
    graph.add_argument("capture")
    graph.add_argument("--output", required=True)
    checkpoint = commands.add_parser("compare-checkpoint", help="比较安全张量 checkpoint 数据集")
    checkpoint.add_argument("reference")
    checkpoint.add_argument("candidate")
    checkpoint.add_argument("--atol", type=float)
    checkpoint.add_argument("--rtol", type=float)
    checkpoint.add_argument("--output", required=True)
    conversion = commands.add_parser("convert-checkpoint", help="将可信本地 checkpoint 转换为离线张量数据集")
    conversion.add_argument("source")
    conversion.add_argument("--output", required=True)
    conversion.add_argument("--key")
    conversion.add_argument("--layout", help="明确的分片布局 JSON")
    conversion.add_argument("--max-bytes", type=int, default=1024**3)
    layout = commands.add_parser("reconstruct", help="校验并重组连续参数或梯度分片")
    layout.add_argument("capture")
    layout.add_argument("--output", required=True)
    layout.add_argument("--parameter")
    layout.add_argument("--step", type=int)
    layout.add_argument("--max-bytes", type=int, default=1024**3)
    axes = commands.add_parser("reconstruct-layout", help="按显式全局形状和坐标重组矩形分片")
    axes.add_argument("capture")
    axes.add_argument("--layout", required=True)
    axes.add_argument("--output", required=True)
    axes.add_argument("--max-bytes", type=int, default=1024**3)
    precheck = commands.add_parser("precheck", help="固定输入与梯度种子的算子前反向预检")
    precheck.add_argument("op")
    precheck.add_argument("--device", choices=("cpu", "musa", "cuda"), default="cpu")
    precheck.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    precheck.add_argument("--seed", type=int, default=0)
    precheck.add_argument("--forward-only", action="store_true")
    precheck.add_argument("--output", required=True)
    precheck.add_argument("--atol", type=float)
    precheck.add_argument("--rtol", type=float)
    captured = commands.add_parser("precheck-captured", help="用原采集输入和显式新梯度种子预检")
    captured.add_argument("rank_dir")
    captured.add_argument("--event-id", required=True, type=int)
    captured.add_argument("--device", choices=("cpu", "musa", "cuda"), default="cpu")
    captured.add_argument("--gradient-seed", type=int, default=0)
    captured.add_argument("--forward-only", action="store_true")
    captured.add_argument("--output", required=True)
    framework = commands.add_parser("framework-info", help="检查框架安装和后端支持证据")
    framework.add_argument("name")
    importer = commands.add_parser("import-tensors", help="导入其他框架导出的安全 NumPy 张量清单")
    importer.add_argument("manifest")
    importer.add_argument("--framework", required=True)
    importer.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        exit_code = 0
        if args.command == "inspect":
            from .compare import inspect_run
            result = inspect_run(args.capture)
        elif args.command == "compare":
            from .compare import compare_runs
            full = compare_runs(args.reference, args.candidate, atol=args.atol, rtol=args.rtol,
                                max_ulp=args.max_ulp, rules=args.rules, mapping=args.mapping, output=args.output)
            result = {key: value for key, value in full.items() if key != "records"}
            exit_code = {"pass": 0, "fail": 1, "incomplete": 2}[full["overall"]]
        elif args.command == "report":
            from .report import render_report
            with Path(args.comparison).open(encoding="utf-8") as stream:
                comparison = json.load(stream)
            result = {"report": render_report(comparison, args.output)}
        elif args.command == "replay":
            from .replay import replay_call
            result = replay_call(args.rank_dir, args.event_id, device=args.device,
                                 reference_dtype=args.reference_dtype, atol=args.atol, rtol=args.rtol)
            if args.output:
                from .storage import atomic_json
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(target, result)
            exit_code = {"pass": 0, "mismatch": 1, "unsupported": 2, "error": 2}[result["status"]]
        elif args.command == "check-config":
            from .analysis import compare_configs
            result = compare_configs(args.reference, args.candidate, output=args.output)
        elif args.command == "suggest-mapping":
            from .analysis import suggest_mapping
            result = suggest_mapping(args.reference, args.candidate, output=args.output)
        elif args.command == "trend":
            from .analysis import build_trends
            result = build_trends(args.capture, output=args.output)
        elif args.command == "graph":
            from .graph import analyze_graph
            result = analyze_graph(args.capture, output=args.output)
        elif args.command == "compare-checkpoint":
            from .checkpoint import compare_checkpoints
            options = {k: getattr(args, k) for k in ("atol", "rtol") if getattr(args, k) is not None}
            result = compare_checkpoints(args.reference, args.candidate, output=args.output, **options)
        elif args.command == "convert-checkpoint":
            from .checkpoint import convert_checkpoint
            result = convert_checkpoint(args.source, args.output, key=args.key, layout=args.layout, max_bytes=args.max_bytes)
        elif args.command == "reconstruct":
            from .layout import reconstruct_shards
            result = reconstruct_shards(args.capture, args.output, parameter=args.parameter, step=args.step, max_bytes=args.max_bytes)
        elif args.command == "reconstruct-layout":
            from .layout_axes import reconstruct_axes
            result = reconstruct_axes(args.capture, args.layout, args.output, max_bytes=args.max_bytes)
        elif args.command == "precheck":
            from .precheck import run_precheck
            options = {k: getattr(args, k) for k in ("atol", "rtol") if getattr(args, k) is not None}
            result = run_precheck(args.op, args.output, device=args.device, dtype=args.dtype,
                                  seed=args.seed, backward=not args.forward_only, **options)
        elif args.command == "precheck-captured":
            from .precheck import precheck_captured
            result = precheck_captured(args.rank_dir, args.event_id, args.output, device=args.device,
                                       backward=not args.forward_only, gradient_seed=args.gradient_seed)
        elif args.command == "framework-info":
            from .frameworks import inspect_framework
            result = inspect_framework(args.name)
        elif args.command == "import-tensors":
            from .frameworks import import_tensor_bundle
            result = import_tensor_bundle(args.manifest, args.output, framework=args.framework)
        if args.command not in ("inspect", "compare", "report", "replay"):
            status = result.get("overall", result.get("status", "pass"))
            exit_code = {"pass": 0, "equal": 0, "fail": 1, "mismatch": 1, "different": 1,
                         "incomplete": 2, "partial": 2, "unknown": 2, "unsupported": 2, "error": 2}.get(status, 0)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"musaProbe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
