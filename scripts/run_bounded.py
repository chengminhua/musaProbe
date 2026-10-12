#!/usr/bin/env python3
"""Record a bounded experiment in an already resource-limited task container.

Usage: python scripts/run_bounded.py --lab /data/task --name check -- python ...
The caller supplies Docker CPU/memory/PID/GPU limits. This utility records the
visible cgroup limits, command, source hashes, package versions and exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--budget-gib", type=int, default=200)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    lab = Path(args.lab).resolve()
    if not lab.is_dir() or lab == Path("/"):
        parser.error("lab must be an existing task directory")
    if not args.name or Path(args.name).name != args.name or args.name.startswith("."):
        parser.error("name must be a simple experiment name")
    if not 0 < args.timeout <= 2700 or not 0 < args.budget_gib <= 200:
        parser.error("timeout must be <=2700 seconds; budget must be <=200 GiB")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    free = shutil.disk_usage(lab).free
    allocated = sum(path.stat().st_size for path in lab.rglob("*") if path.is_file() and not path.is_symlink())
    if free < 500 * 1024**3 or allocated >= args.budget_gib * 1024**3:
        parser.error("disk floor or task data budget would be violated")
    output = lab / "evidence" / args.name
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ("src", "examples", "scripts") for path in (root / folder).rglob("*.py")}
    packages = {}
    for name in ("torch", "torch_musa", "numpy", "transformer-engine", "torchada", "triton", "vllm", "sglang", "verl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cgroups = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "pids.max", "memory.swap.max"):
        path = Path("/sys/fs/cgroup") / name
        cgroups[name] = path.read_text().strip() if path.exists() else None
    record = {"command": command, "cwd": str(Path.cwd()), "image_id": args.image_id,
              "timeout_seconds": args.timeout, "packages": packages, "source_sha256": hashes,
              "cgroup_limits": cgroups, "disk_free_before": free, "task_bytes_before": allocated,
              "started_unix": time.time(), "gpu_visibility": os.environ.get("MTHREADS_VISIBLE_DEVICES")}
    record["runtime_controls"] = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "MUSAPROBE_INTRAOP_THREADS", "MUSAPROBE_INTEROP_THREADS", "MAX_JOBS",
        "GLOO_SOCKET_IFNAME", "MCCL_SOCKET_IFNAME", "MCCL_SOCKET_NTHREADS", "MCCL_NSOCKS_PERTHREAD")}
    with (output / "run.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record["exit_code"] = process.wait(timeout=args.timeout)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            record["exit_code"] = 124
    record["elapsed_seconds"] = time.time() - record["started_unix"]
    record["disk_free_after"] = shutil.disk_usage(lab).free
    (output / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: record[k] for k in ("exit_code", "timed_out", "elapsed_seconds")}))
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
