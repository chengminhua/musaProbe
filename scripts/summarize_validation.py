#!/usr/bin/env python3
"""Summarize explicitly selected validation runs without starting training.

Inputs are read-only. The output directory must be outside runs/ and captures/.
Only the Python standard library is required, including on the remote worker.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Load the sibling script by its own path, so both direct CLI execution and
# importlib-based callers use exactly the same scanner without PYTHONPATH setup.
_VALIDATOR_SPEC = importlib.util.spec_from_file_location("_musaprobe_runtime_validator", Path(__file__).with_name("validate_remote_run.py"))
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)
inspect_runtime_errors = _VALIDATOR.inspect_runtime_errors


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_ITERATION = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_FIELDS = {
    "elapsed_ms": re.compile(r"elapsed\s+time\s+per\s+iteration\s*\(ms\)\s*:\s*(" + _NUMBER + ")", re.IGNORECASE),
    "loss": re.compile(r"(?:\blm\s+loss|\blanguage\s+model\s+loss|(?<![\w ])loss)\s*:\s*(" + _NUMBER + ")", re.IGNORECASE),
    "grad_norm": re.compile(r"\bgrad(?:ient)?[ _-]*norm\s*:\s*(" + _NUMBER + ")", re.IGNORECASE),
    "learning_rate": re.compile(r"\blearning\s+rate\s*:\s*(" + _NUMBER + ")", re.IGNORECASE),
}


def _read_text(path, warnings, *, required=False):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        if required or not isinstance(error, FileNotFoundError):
            warnings.append(f"{path.name}: {type(error).__name__}: {error}")
        return None


def _read_json(path, warnings, *, required=False):
    raw = _read_text(path, warnings, required=required)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except ValueError as error:
        warnings.append(f"{path.name}: {error}")
        return {}


def parse_training_log(path):
    """Return every distinct logged iteration and diagnostics for duplicates."""
    rows, warnings, profiles = [], [], set()
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except OSError as error:
        return [], [f"train.log: {type(error).__name__}: {error}"], None
    with stream:
        for line_number, line in enumerate(stream, 1):
            header = re.search(r"Starting run=\S+\s+profile=(baseline|statistics|tensor)\b", line)
            if header:
                profiles.add(header.group(1))
            iteration = _ITERATION.search(line)
            if iteration is None or _FIELDS["elapsed_ms"].search(line) is None:
                continue
            count = int(iteration.group(1))
            if count < 1:
                warnings.append(f"line {line_number}: invalid one-based iteration {count}")
                continue
            row = {"iteration": count, "step": count - 1,
                   "planned_iterations": int(iteration.group(2)), "log_line": line_number,
                   "nonfinite_metrics": {}}
            for field, pattern in _FIELDS.items():
                match = pattern.search(line)
                row[field] = None
                if match:
                    value = float(match.group(1))
                    if math.isfinite(value):
                        row[field] = value
                    else:
                        row["nonfinite_metrics"][field] = match.group(1)
            if row["elapsed_ms"] is not None and row["elapsed_ms"] < 0:
                warnings.append(f"line {line_number}: negative elapsed time excluded from averages")
                row["elapsed_ms"] = None
            rows.append(row)
    unique = {}
    conflicting = set()
    for row in rows:
        step = row["step"]
        if step not in unique:
            unique[step] = row
        else:
            comparable = {key: value for key, value in row.items() if key not in {"log_line", "duplicate_log_lines"}}
            previous = {key: value for key, value in unique[step].items() if key not in {"log_line", "duplicate_log_lines"}}
            if comparable != previous:
                conflicting.add(step)
            else:
                unique[step].setdefault("duplicate_log_lines", []).append(row["log_line"])
    for step in sorted(conflicting):
        unique[step]["ambiguous_log_records"] = True
        warnings.append(f"step {step}: conflicting duplicate log records; excluded from timing comparisons")
    if len(profiles) > 1:
        warnings.append("train.log contains more than one profile header")
    profile = next(iter(profiles)) if len(profiles) == 1 else None
    return [unique[step] for step in sorted(unique)], warnings, profile


def _window(rows, selected):
    rows = [row for row in rows if row["step"] in selected and not row.get("ambiguous_log_records")]
    valid = [row for row in rows if row.get("elapsed_ms") is not None]
    values = [row["elapsed_ms"] for row in valid]
    return {"steps": [row["step"] for row in valid], "count": len(values),
            "mean_ms": statistics.mean(values) if values else None,
            "min_ms": min(values) if values else None, "max_ms": max(values) if values else None}


def summarize_rank(directory):
    warnings = []
    manifest = _read_json(directory / "manifest.json", warnings, required=True)
    states, kinds, phases, steps, by_step = Counter(), Counter(), Counter(), Counter(), {}
    nonfinite = {"events": 0, "nan_count": 0, "posinf_count": 0, "neginf_count": 0, "examples": []}
    event_count, malformed = 0, 0
    try:
        stream = (directory / "events.jsonl").open(encoding="utf-8", errors="replace")
    except OSError as error:
        warnings.append(f"events.jsonl: {type(error).__name__}: {error}")
        stream = None
    if stream is not None:
        with stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("expected event object")
                    step = str(event.get("step", "missing"))
                    state, kind, phase = (str(event.get(key, "missing")) for key in ("state", "kind", "phase"))
                    stats = event.get("stats") or {}
                    if not isinstance(stats, dict):
                        raise ValueError("event stats must be an object")
                    counts = {key: stats.get(key, 0) for key in ("nan_count", "posinf_count", "neginf_count")}
                    if any(type(value) is not int or value < 0 for value in counts.values()):
                        raise ValueError("nonfinite counts must be nonnegative integers")
                except (ValueError, TypeError) as error:
                    malformed += 1
                    if malformed <= 5:
                        warnings.append(f"events.jsonl line {line_number}: {error}")
                    continue
                event_count += 1
                states[state] += 1
                kinds[kind] += 1
                phases[phase] += 1
                steps[step] += 1
                group = by_step.setdefault(step, {"events": 0, "states": Counter(), "kinds": Counter(), "phases": Counter(), "nonfinite_events": 0})
                group["events"] += 1
                group["states"][state] += 1
                group["kinds"][kind] += 1
                group["phases"][phase] += 1
                if any(counts.values()):
                    nonfinite["events"] += 1
                    group["nonfinite_events"] += 1
                    for key, value in counts.items():
                        nonfinite[key] += value
                    if len(nonfinite["examples"]) < 8:
                        nonfinite["examples"].append({key: event.get(key) for key in
                                                       ("event_id", "step", "phase", "module", "kind", "tensor_path")})
    actual_bytes = 0
    for path in (directory / "tensors").glob("*.npy"):
        try:
            if not path.is_symlink() and path.is_file():
                actual_bytes += path.stat().st_size
        except OSError as error:
            warnings.append(f"tensor stat: {error}")
    expected_count = manifest.get("events_count")
    if manifest.get("status") == "complete" and expected_count != event_count:
        warnings.append(f"complete manifest events_count={expected_count} differs from readable events={event_count}")
    metadata = manifest.get("metadata") or {}
    return {"rank": manifest.get("rank", directory.name), "status": manifest.get("status", "unknown"),
            "events": event_count, "malformed_lines": malformed, "states": dict(states), "kinds": dict(kinds),
            "phases": dict(phases), "steps": dict(steps), "by_step": by_step,
            "manifest_tensor_bytes": manifest.get("tensor_bytes"), "tensor_file_bytes": actual_bytes,
            "nonfinite": nonfinite,
            "parallel_coordinates": metadata.get("parallel_coordinates") if isinstance(metadata, dict) else None,
            "warnings": warnings}


def summarize_run(base, run_id, default_selected_steps):
    run_dir = base / "runs" / run_id
    warnings = []
    status = _read_text(run_dir / "status", warnings, required=True) or "unknown"
    raw_exit = _read_text(run_dir / "exit_code", warnings)
    try:
        exit_code = None if raw_exit is None else int(raw_exit)
    except ValueError:
        exit_code = None
        warnings.append("exit_code is not an integer")
    config = _read_json(run_dir / "probe_config.json", warnings)
    rows, log_warnings, profile = parse_training_log(run_dir / "train.log")
    warnings.extend(log_warnings)
    log_present = (run_dir / "train.log").is_file()
    runtime_scan_complete = False
    runtime_error_count, runtime_errors = 0, []
    if log_present:
        try:
            runtime_error_count, runtime_errors = inspect_runtime_errors(run_dir / "train.log")
            runtime_scan_complete = True
        except OSError as error:
            warnings.append(f"runtime error scan failed: {error}")
    if runtime_error_count:
        warnings.append(f"Detected {runtime_error_count} runtime-error log lines; exit code 0 does not establish a clean run.")
    config_profile = config.get("mode")
    if profile is None and config_profile in ("statistics", "tensor"):
        profile = config_profile
    elif profile in ("statistics", "tensor") and config_profile not in (None, profile):
        warnings.append("log profile differs from probe_config mode")
    selected = config.get("steps", default_selected_steps)
    if selected is None:
        selected = [row["step"] for row in rows]
    if not isinstance(selected, list) or any(type(step) is not int or step < 0 for step in selected):
        warnings.append("invalid config steps; timing windows use CLI default steps")
        selected = default_selected_steps
    selected = sorted(set(selected))
    observed = {row["step"] for row in rows}
    captures = base / "captures" / run_id
    ranks = [summarize_rank(directory) for directory in sorted(captures.glob("rank_*")) if directory.is_dir()]
    return {"run_id": run_id, "profile": profile or "unknown", "status": status, "exit_code": exit_code,
            "log_present": log_present, "runtime_scan_complete": runtime_scan_complete,
            "runtime_error_count": runtime_error_count, "runtime_errors": runtime_errors,
            "clean_runtime_exit": bool(status == "completed" and exit_code == 0 and runtime_scan_complete and runtime_error_count == 0),
            "injected": (run_dir / "injection.json").is_file(), "selected_capture_steps": selected,
            "capture_steps_source": "probe_config" if "steps" in config else "CLI comparison default",
            "training_steps": rows,
            "timing": {"step0_initialization": _window(rows, {0}),
                       "selected_steps_after_step0": _window(rows, set(selected) - {0}),
                       "non_capture_steps_after_step0": _window(rows, observed - set(selected) - {0}),
                       "all_steps_after_step0": _window(rows, observed - {0})},
            "captures": {"ranks": ranks, "events": sum(rank["events"] for rank in ranks),
                         "tensor_file_bytes": sum(rank["tensor_file_bytes"] for rank in ranks),
                         "nonfinite_events": sum(rank["nonfinite"]["events"] for rank in ranks)},
            "warnings": warnings}


def calculate_overheads(runs):
    baselines = [run for run in runs if run["profile"] == "baseline" and run["status"] == "completed" and
                 run["exit_code"] == 0 and not run["injected"]]
    result = []
    for run in runs:
        if (run["profile"] not in ("statistics", "tensor") or run["status"] != "completed" or
                run["exit_code"] != 0 or run["injected"]):
            continue
        for baseline in baselines:
            source = {row["step"]: row for row in baseline["training_steps"]
                      if row["elapsed_ms"] is not None and not row.get("ambiguous_log_records")}
            target = {row["step"]: row for row in run["training_steps"]
                      if row["elapsed_ms"] is not None and not row.get("ambiguous_log_records")}
            common = (set(source) & set(target)) - {0}
            selected = set(run["selected_capture_steps"])
            windows = {}
            for name, steps in (("selected_steps_after_step0", common & selected),
                                ("non_capture_steps_after_step0", common - selected),
                                ("all_steps_after_step0", common)):
                first = _window(list(source.values()), steps)
                second = _window(list(target.values()), steps)
                base_mean, measured_mean = first["mean_ms"], second["mean_ms"]
                windows[name] = {"steps": sorted(steps), "count": len(steps),
                                 "baseline_mean_ms": base_mean, "capture_mean_ms": measured_mean,
                                 "delta_ms": measured_mean - base_mean if steps else None,
                                 "overhead_percent": (measured_mean / base_mean - 1) * 100 if steps and base_mean > 0 else None}
            result.append({"run_id": run["run_id"], "baseline_run_id": baseline["run_id"],
                           "profile": run["profile"], "windows": windows,
                           "has_runtime_errors": bool(run["runtime_error_count"] or baseline["runtime_error_count"]),
                           "runtime_error_counts": {"run": run["runtime_error_count"], "baseline": baseline["runtime_error_count"]},
                           "eligible_for_clean_run_comparison": bool(run["clean_runtime_exit"] and baseline["clean_runtime_exit"]),
                           "interpretation": "Short diagnostic timing only; runs with runtime errors are not clean validation or formal benchmark evidence."})
    return result


def build_summary(base, run_ids, selected_steps=(2, 3)):
    runs = [summarize_run(Path(base), run_id, list(selected_steps)) for run_id in run_ids]
    return {"schema_version": 1, "tool": "musaProbe validation summary",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "step_semantics": "Megatron printed iteration is one-based; step=iteration-1. Capture steps are zero-based.",
            "measurement_limitations": [
                "Short validation runs are not a stable performance benchmark; initialization, compilation, caches and run order can dominate.",
                "Step 0 is shown separately and excluded from overhead calculations. Later non-capture steps may still include warmup effects.",
                "Overheads pair completed non-injected runs on exactly matching step indexes; missing or conflicting log rows are excluded.",
                "Zero exit status does not prove a clean run: asynchronous shutdown/collective errors are scanned separately and taint comparisons even when timing rows are available.",
                "Tensor byte counts are disk usage of .npy files, not peak device memory. This script does not infer peak GPU memory.",
                "A completed process or a timing summary does not establish numerical correctness; use musaProbe compare reports for that.",
            ], "runs": runs, "overhead_comparisons": calculate_overheads(runs)}


def _number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(summary):
    lines = ["# musaProbe 短跑验收汇总", "", "日志 iteration 从 1 起；本文 step 从 0 起。**短跑结果只用于记录验收开销，不是稳定性能基准。**",
             "step 0 的初始化单列；选定采集步和其余步分别统计，后续步骤也可能受预热、缓存及运行顺序影响。", "",
             "**退出码 0 仍可能伴随异步通信或关闭错误；运行时错误单独扫描，含错误的耗时仅供诊断。**", "",
             "| run | 模式 | 状态/退出码 | 运行时错误行 | 步数 | step 0 ms | 采集步均值 ms | 非采集步均值 ms | 数据 MiB | 非有限事件 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in summary["runs"]:
        timing = run["timing"]
        lines.append("| " + " | ".join(map(_escape, [
            run["run_id"], run["profile"] + (" + 注入" if run["injected"] else ""),
            f"{run['status']}/{run['exit_code']}", run["runtime_error_count"] if run["runtime_scan_complete"] else "未完成扫描", len(run["training_steps"]),
            _number(timing["step0_initialization"]["mean_ms"]),
            _number(timing["selected_steps_after_step0"]["mean_ms"]),
            _number(timing["non_capture_steps_after_step0"]["mean_ms"]),
            _number(run["captures"]["tensor_file_bytes"] / 1024 ** 2), run["captures"]["nonfinite_events"]])) + " |")
    lines.extend(["", "## 与基线相同步骤的耗时对照", "", "失败、运行中及注入任务不计算开销；step 0 不参与。", "",
                  "| 运行 / 基线 | 窗口 | 实际 step | 基线均值 ms | 当前均值 ms | 差值 ms | 开销 | 数据有效性 |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"])
    labels = {"selected_steps_after_step0": "选定采集步", "non_capture_steps_after_step0": "非采集步", "all_steps_after_step0": "全部后续步"}
    for comparison in summary["overhead_comparisons"]:
        for window, values in comparison["windows"].items():
            lines.append("| " + " | ".join(map(_escape, [
                comparison["run_id"] + " / " + comparison["baseline_run_id"], labels[window],
                ",".join(map(str, values["steps"])) or "无匹配步骤", _number(values["baseline_mean_ms"]),
                _number(values["capture_mean_ms"]), _number(values["delta_ms"]),
                "—" if values["overhead_percent"] is None else _number(values["overhead_percent"]) + "%",
                "含运行时错误，仅诊断" if comparison["has_runtime_errors"] else
                "短跑，未检出运行时错误" if comparison["eligible_for_clean_run_comparison"] else "日志未完成扫描，仅诊断"])) + " |")
    if not summary["overhead_comparisons"]:
        lines.extend(["", "尚无完成且无注入的基线/采集组合，未计算开销。"])
    for run in summary["runs"]:
        lines.extend(["", f"## {_escape(run['run_id'])}", "",
                      "选定采集 step：" + ", ".join(map(str, run["selected_capture_steps"])) + "。"])
        if run["runtime_errors"]:
            lines.extend(["", f"运行时错误共 {run['runtime_error_count']} 行，以下列出前 {min(10, len(run['runtime_errors']))} 行；该任务不能仅凭 completed/exit 0 认定验收通过。", ""])
            for error in run["runtime_errors"][:10]:
                lines.append(f"- 日志行 {error['line']}（{_escape(error['category'])}）：{_escape(error['message'])}")
        lines.extend(["",
                      "| step | elapsed ms | loss | grad norm | LR |", "| ---: | ---: | ---: | ---: | ---: |"])
        for row in run["training_steps"]:
            lines.append(f"| {row['step']} | {_number(row['elapsed_ms'])} | {_number(row['loss'], 6)} | {_number(row['grad_norm'], 6)} | {_number(row['learning_rate'], 9)} |")
            if row["nonfinite_metrics"]:
                lines.extend(["", f"step {row['step']} 非有限日志指标：`{json.dumps(row['nonfinite_metrics'], ensure_ascii=False)}`", ""])
        lines.extend(["", "| rank | 状态 | 事件 | state 计数 | kind 计数 | phase 计数 | step 计数 | MiB | 非有限事件 |",
                      "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |"])
        warnings = list(run["warnings"])
        for rank in run["captures"]["ranks"]:
            counter = lambda name: ", ".join(f"{key}:{value}" for key, value in sorted(rank[name].items())) or "—"
            lines.append("| " + " | ".join(map(_escape, [rank["rank"], rank["status"], rank["events"],
                counter("states"), counter("kinds"), counter("phases"), counter("steps"),
                _number(rank["tensor_file_bytes"] / 1024 ** 2), rank["nonfinite"]["events"]])) + " |")
            warnings.extend(f"rank {rank['rank']}: {warning}" for warning in rank["warnings"])
        if warnings:
            lines.extend(["", "解析与覆盖提示：", ""] + ["- " + _escape(warning) for warning in warnings])
    lines.extend(["", "逐步、逐 rank、逐 state/kind/phase 的完整计数见同目录 `summary.json`。磁盘字节数不是峰值显存；数值正确性仍需查看 compare 报告。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/musaprobe")
    parser.add_argument("--run-id", action="append", required=True, help="Repeat for each explicit run to summarize")
    parser.add_argument("--selected-steps", default="2,3", help="Zero-based comparison window when no probe config is present")
    parser.add_argument("--output", required=True, help="Summary output directory, outside input runs/ and captures/")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) for value in args.run_id):
        parser.error("each run-id must be a safe single directory name")
    if not re.fullmatch(r"\d+(,\d+)*", args.selected_steps):
        parser.error("selected-steps must be comma-separated nonnegative integers")
    base, output = Path(args.base).resolve(), Path(args.output).resolve()
    if any(output.is_relative_to((base / name).resolve()) for name in ("runs", "captures")):
        parser.error("output must be outside the read-only runs/ and captures/ input trees")
    summary = build_summary(base, list(dict.fromkeys(args.run_id)), sorted({int(step) for step in args.selected_steps.split(",")}))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"runs": len(summary["runs"]), "json": str(output / "summary.json"), "markdown": str(output / "summary.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
