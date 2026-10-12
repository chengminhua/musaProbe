#!/usr/bin/env python3
"""Read-only post-run inspection, optionally rehashing the protected original."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect_runtime_errors(log_path):
    """Keep real runtime failures even when torchrun returned zero at shutdown.

    Warnings and configuration values containing the word 'error' are not errors.
    The matchers target C++ severity records, Python exceptions, and runtime
    logging/collective failures, not arbitrary occurrences of that word.
    """
    errors = []
    count = 0
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.rstrip()
            message = re.sub(r"^(?:\[[^\]]+\]:)+\s*", "", line)
            category = None
            if re.search(r"\[[EF]\d{4}\s", message):
                category = "cpp_error"
            elif (re.search(r"\[W\d{4}\s", message)
                  or re.search(r"\b(?:\w*Warning|WARNING):", message)):
                continue
            elif message.startswith("Traceback (most recent call last):"):
                category = "python_traceback"
            elif re.match(r"(?:[\w.]*Error|[\w.]*Exception|KeyboardInterrupt|SystemExit):", message):
                category = "python_exception"
            elif re.search(r"\b(?:MCCL|NCCL|CUDA|MUSA) (?:error|ERROR)\b|unhandled system error", message):
                category = "device_runtime_error"
            elif re.search(r"(?:^|\s|\[)(?:ERROR|FATAL)(?=\s|:|\])", message):
                category = "logged_error"
            elif re.match(r"(?:Error|Fatal error):", message):
                category = "logged_error"
            if category is not None:
                count += 1
                if len(errors) < 100:
                    errors.append({"line": number, "category": category, "message": line[:1600]})
    return count, errors


def digest(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/musaprobe")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--require-capture", action="store_true")
    args = parser.parse_args()
    base = Path(args.base)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be one directory name")
    run = base / "runs" / args.run_id
    report = {"run": str(run), "status": (run / "status").read_text().strip(),
              "exit_code": int((run / "exit_code").read_text()), "ranks": [], "source_mismatches": [],
              "source_added_files": []}
    log = run / "train.log"
    report["log_present"] = log.is_file()
    report["runtime_error_count"], report["runtime_errors"] = inspect_runtime_errors(log) if log.is_file() else (0, [])
    for manifest_file in sorted((base / "captures" / args.run_id).glob("rank_*/manifest.json")):
        manifest = json.loads(manifest_file.read_text())
        events = [json.loads(line) for line in (manifest_file.parent / "events.jsonl").read_text().splitlines()]
        report["ranks"].append({
            "rank": manifest["rank"], "status": manifest["status"], "events": len(events),
            "steps": sorted({item["step"] for item in events}),
            "phases": sorted({item["phase"] for item in events}),
            "microbatches": sorted({item["microbatch"] for item in events if item["microbatch"] is not None}),
            "states": manifest["states"], "tensor_bytes": manifest["tensor_bytes"],
            "collector_errors": manifest["metadata"].get("collector_errors", []),
            "parallel_coordinates": manifest["metadata"].get("parallel_coordinates"),
            "nonfinite_events": sum(bool(item.get("stats", {}).get("nan_count", 0)
                or item.get("stats", {}).get("posinf_count", 0)
                or item.get("stats", {}).get("neginf_count", 0)) for item in events),
        })
    if args.verify_source:
        audit = json.loads((base / "audit/source_manifest.json").read_text())
        source = Path(audit["source"])
        for relative, expected in audit["source_files"].items():
            path = source / relative
            if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                report["source_mismatches"].append(relative)
        # Detect newly created caches/bytecode as well as modified original files.
        current_files = set()
        for relative in ("code", "scripts", "tokenizer", "data/deepseekv2_full_epoch", "hostfile"):
            item = source / relative
            paths = item.rglob("*") if item.is_dir() else [item]
            current_files.update(str(path.relative_to(source)) for path in paths if path.is_file())
        report["source_added_files"] = sorted(current_files - set(audit["source_files"]))
        report["original_files_verified"] = len(audit["source_files"])
    passed = (report["exit_code"] == 0 and report["status"] == "completed"
              and not report["source_mismatches"] and not report["source_added_files"] and report["log_present"]
              and report["runtime_error_count"] == 0)
    passed = passed and not any(rank["states"].get("error", 0) or rank["collector_errors"] for rank in report["ranks"])
    if args.require_capture:
        active = [rank for rank in report["ranks"] if rank["events"]]
        passed = passed and bool(active) and all(rank["status"] == "complete" for rank in report["ranks"])
        passed = passed and all({"forward", "backward", "optimizer"} <= set(rank["phases"]) for rank in active)
    report["inspection_passed"] = bool(passed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
