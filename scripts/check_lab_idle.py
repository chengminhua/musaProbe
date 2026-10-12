#!/usr/bin/env python3
"""Read-only host guard before an isolated GPU experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--budget-gib", type=int, default=200)
    args = parser.parse_args()
    lab = Path(args.lab).resolve(strict=True)
    if args.budget_gib <= 0 or args.budget_gib > 200:
        parser.error("budget must be in 1..200 GiB")
    free = shutil.disk_usage(lab).free
    size = sum(p.stat().st_size for p in lab.rglob("*") if p.is_file() and not p.is_symlink())
    gpu = json.loads(subprocess.check_output(["mthreads-gmi", "-q", "--json"], timeout=20))["GPU"]
    busy = [g["Index"] for g in gpu if g["FB Memory Usage"]["Used"] != "0MiB" or g["Utilization"]["Gpu"] != "0%"]
    if not gpu:
        raise RuntimeError("No GPU devices reported; do not launch")
    evidence = {"time": time.time(), "gpu": gpu, "busy": busy, "disk_free": free, "task_bytes": size,
                "allowed": not busy and free >= 500 * 1024**3 and size < args.budget_gib * 1024**3}
    checks = lab / "host-checks"
    checks.mkdir(exist_ok=True)
    (checks / f"{time.time_ns()}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("busy", "disk_free", "task_bytes", "allowed")}))
    return 0 if evidence["allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
