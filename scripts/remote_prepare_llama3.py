#!/usr/bin/env python3
"""Run inside train container: copy inputs and minimally adapt the launch script.

Does not import the training framework or launch training. Existing destination
refuses reuse, so every setup is auditable and does not overwrite an experiment.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess

DEFAULT_BASE = "/data/musaprobe"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest_tree(root, relative_paths):
    result = {}
    for relative in relative_paths:
        item = root / relative
        files = sorted(item.rglob("*")) if item.is_dir() else [item]
        for path in files:
            if path.is_file():
                result[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor, found {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


def patch_training_script(original, source, target, base, *, original_work_home=None):
    """Patch only the documented single-node Llama3 launcher layout."""
    assignments = re.findall(r"^WORK_HOME=(.*)$", original, re.MULTILINE)
    if len(assignments) != 1:
        raise ValueError("Expected exactly one literal WORK_HOME assignment")
    values = shlex.split(assignments[0])
    declared_source = source if original_work_home is None else Path(original_work_home)
    if len(values) != 1 or Path(values[0]).resolve() != declared_source:
        raise ValueError("WORK_HOME must be a literal path matching --source")
    patched = replace_once(original, "WORK_HOME=" + assignments[0], "WORK_HOME=" + shlex.quote(str(target)))
    for name in ("MASTER_ADDR", "MASTER_PORT"):
        pattern = rf"^{name}=.*$"
        if len(re.findall(pattern, patched, re.MULTILINE)) != 1:
            raise ValueError(f"Expected exactly one {name} assignment; unsupported launcher")
        patched = re.sub(pattern, f'{name}=${{MUSAPROBE_{name}:?set {name} through remote_launch_llama3.sh}}',
                         patched, flags=re.MULTILINE)
    patched = replace_once(patched, "RUN_ID=${1:?run id required}",
        'RUN_ID=${1:?run id required}\nPROBE_BASE=' + shlex.quote(str(base)) + '\n'
        'PROBE_PACKAGE=${MUSAPROBE_PACKAGE_ROOT:-$PROBE_BASE/package}\n'
        'EXIT_INTERVAL=${MUSAPROBE_EXIT_INTERVAL:?must be 2 or 8}\n'
        '[[ $EXIT_INTERVAL == 2 || $EXIT_INTERVAL == 8 ]] || { echo "exit interval must be 2 or 8" >&2; exit 2; }')
    replacements = [
        ("TB_PATH=$WORK_HOME/tboard/$EXPNAME", 'TB_PATH=$PROBE_BASE/runs/$RUN_ID/tensorboard'),
        ("RANK_LOG_PATH=$WORK_HOME/output/deepseekv2_full/$RUN_ID/ranks", 'RANK_LOG_PATH=$PROBE_BASE/runs/$RUN_ID/ranks'),
        ('export PYTHONPATH="$MEGATRON_PATH:$PATCH_HOME:${PYTHONPATH:-}"',
         'export PYTHONPATH="$PROBE_PACKAGE/src:$MEGATRON_PATH:$PATCH_HOME:${PYTHONPATH:-}"'),
        ('    --train-samples 94720', '    --train-samples 94720\n    --exit-interval "$EXIT_INTERVAL"'),
        ('    --data-path "$DATA_PATH"', '    --data-path "$DATA_PATH"\n    --data-cache-path "$PROBE_BASE/cache/dataset"'),
        ('cmd=(torchrun "${DISTRIBUTED_ARGS[@]}" "$PATCH_HOME/examples/llama3/pretrain_gpt.py"',
         'cmd=(torchrun "${DISTRIBUTED_ARGS[@]}" "$PROBE_PACKAGE/src/musaprobe/integrations/megatron_entry.py" "$PATCH_HOME/examples/llama3/pretrain_gpt.py"'),
    ]
    for old, new in replacements:
        patched = replace_once(patched, old, new)
    subprocess.run(["bash", "-n"], input=patched, text=True, check=True)
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original, read-only training project directory")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--original-work-home", help="Literal original WORK_HOME when --source is a verified relocated copy")
    parser.add_argument("--master-addr", required=True, help="Explicit rendezvous address on this single-node worker")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--expected-hostname", help="Optional hostname check inside the execution environment")
    args = parser.parse_args()
    if not 1 <= args.master_port <= 65535 or not args.master_addr.strip():
        parser.error("master address must be nonempty and port must be in 1..65535")
    if args.expected_hostname and socket.gethostname() != args.expected_hostname:
        parser.error("Current hostname does not match --expected-hostname")
    if not Path(args.source).is_absolute() or not Path(args.base).is_absolute():
        parser.error("--source and --base must be explicit absolute paths")
    source = Path(args.source).resolve(strict=True)
    base = Path(args.base).resolve()
    if base == Path("/") or base.is_relative_to(source) or source.is_relative_to(base):
        parser.error("Source and output base must be separate, non-overlapping directories")
    target = base / "workspace"
    if target.exists():
        parser.error(f"Destination exists; refusing overwrite: {target}")
    copies = ["code", "scripts", "tokenizer", "data/deepseekv2_full_epoch", "hostfile"]
    if any(not (source / relative).exists() for relative in copies):
        parser.error("Source does not have the documented Llama3 example layout")
    original = (source / "scripts/run_pretrain_llama3_full.sh").read_text()
    if args.original_work_home and not Path(args.original_work_home).is_absolute():
        parser.error("--original-work-home must be an absolute path")
    patched = patch_training_script(original, source, target, base, original_work_home=args.original_work_home)
    before = manifest_tree(source, copies)
    # The complete training dataset is copied to avoid even implicit index
    # cache writes reaching a symlink into the protected source tree.
    target.mkdir(parents=True)
    for relative in copies:
        src, dst = source / relative, target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst)
    for name in ("runs", "captures", "cache", "tmp", "audit", "configs"):
        (base / name).mkdir(exist_ok=True)
    copied = manifest_tree(target, copies)
    if before != copied:
        raise RuntimeError("Copied files differ from their source hashes")
    script = target / "scripts/run_pretrain_llama3_full.sh"
    script.write_text(patched)
    (base / "audit/launch_changes.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile="original/scripts/run_pretrain_llama3_full.sh", tofile="isolated/scripts/run_pretrain_llama3_full.sh")))
    after = manifest_tree(source, copies)
    if before != after:
        raise RuntimeError("Protected source changed during setup")
    report = {
        "adapter_schema": 2, "base": str(base), "prepared_hostname": socket.gethostname(),
        "master_addr": args.master_addr, "master_port": args.master_port,
        "source": str(source), "workspace": str(target), "source_files": before,
        "original_work_home": args.original_work_home,
        "copied_source_hashes_match": True, "original_unchanged_after_setup": True,
        "patched_file": "scripts/run_pretrain_llama3_full.sh",
        "patched_sha256": digest(script),
        "validated_example_configuration": {"num_layers": 32, "tensor_parallel": 1, "pipeline_parallel": 2,
            "data_parallel": 4, "micro_batch_size": 1, "global_batch_size": 512,
            "sequence_length": 4096, "train_samples": 94720, "warmup_samples": 4096,
            "lr_decay_samples": 94720, "bf16": True, "transformer_engine_fp8": "hybrid"},
    }
    (base / "audit/source_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    subprocess.run(["bash", "-n", str(script)], check=True)
    print(json.dumps({"prepared": str(target), "source_files": len(before), "original_unchanged": True,
                      "manifest": str(base / "audit/source_manifest.json"), "training_started": False}))


if __name__ == "__main__":
    main()
