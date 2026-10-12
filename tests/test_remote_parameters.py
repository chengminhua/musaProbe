"""Parameterized preparation/dry-run tests: no device runtime or training starts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts/remote_prepare_llama3.py"
LAUNCH = ROOT / "scripts/remote_launch_llama3.sh"


class RemoteParametersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="musaprobe-remote-parameters-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original project"
        self.base = self.root / "isolated project"
        for relative in ("code/Megatron-LM", "code/megatron-lm-musa-patch", "scripts",
                         "tokenizer", "data/deepseekv2_full_epoch"):
            (self.source / relative).mkdir(parents=True)
        (self.source / "hostfile").write_text("local-test-only\n")
        (self.source / "tokenizer/tokenizer.json").write_text("{}\n")
        (self.source / "data/deepseekv2_full_epoch/sample.bin").write_bytes(b"test-data")
        self.script = self.source / "scripts/run_pretrain_llama3_full.sh"
        self.script.write_text("""#!/usr/bin/env bash
set -euo pipefail
WORK_HOME=SOURCE_LITERAL
RUN_ID=${1:?run id required}
MASTER_ADDR=old-example-host
MASTER_PORT=12345
MEGATRON_PATH=$WORK_HOME/code/Megatron-LM
PATCH_HOME=$WORK_HOME/code/megatron-lm-musa-patch
EXPNAME=example
DATA_PATH=$WORK_HOME/data/deepseekv2_full_epoch/sample
TB_PATH=$WORK_HOME/tboard/$EXPNAME
RANK_LOG_PATH=$WORK_HOME/output/deepseekv2_full/$RUN_ID/ranks
export PYTHONPATH="$MEGATRON_PATH:$PATCH_HOME:${PYTHONPATH:-}"
DISTRIBUTED_ARGS=(--nnodes 1 --nproc_per_node 8 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
TRAIN_ARGS=(
    --train-samples 94720
    --data-path "$DATA_PATH"
)
cmd=(torchrun "${DISTRIBUTED_ARGS[@]}" "$PATCH_HOME/examples/llama3/pretrain_gpt.py" "${TRAIN_ARGS[@]}")
if [[ ${FULL_DRY_RUN:-0} == 1 ]]; then
    printf '%s\\n' "${cmd[@]}"
    exit 0
fi
echo 'TEST REFUSES ANY TRAINING EXECUTION' >&2
exit 77
""".replace("SOURCE_LITERAL", shlex.quote(str(self.source))))
        package = self.base / "package/src/musaprobe/integrations"
        package.mkdir(parents=True)
        (package / "megatron_entry.py").write_text("# dry-run placeholder, never executed\n")

    def prepare(self, *extra):
        return subprocess.run([sys.executable, str(PREPARE), "--source", str(self.source),
            "--base", str(self.base), "--master-addr", "127.0.0.1", "--master-port", "29517", *extra],
            text=True, capture_output=True)

    def launch(self, *extra):
        return subprocess.run(["bash", str(LAUNCH), "--base", str(self.base),
            "--master-addr", "127.0.0.1", "--master-port", "29517", "--run-id", "dry_run", *extra],
            text=True, capture_output=True)

    def test_prepare_and_dry_run_use_explicit_endpoint_and_quoted_paths(self):
        original = hashlib.sha256(self.script.read_bytes()).hexdigest()
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.base / "audit/source_manifest.json").read_text())
        self.assertEqual(manifest["adapter_schema"], 2)
        self.assertEqual(manifest["master_addr"], "127.0.0.1")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("127.0.0.1\n--master_port\n29517", result.stdout)
        self.assertIn(str(self.base / "workspace"), result.stdout)
        self.assertNotIn("old-example-host", result.stdout)
        self.assertIn("no training started", result.stdout)
        self.assertFalse((self.base / "runs/dry_run").exists())
        self.assertEqual(hashlib.sha256(self.script.read_bytes()).hexdigest(), original)

    def test_missing_source_and_master_parameters_are_rejected(self):
        result = subprocess.run([sys.executable, str(PREPARE)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--source", result.stderr)
        result = subprocess.run(["bash", str(LAUNCH), "--run-id", "dry_run"], capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Explicit master", result.stderr)

    def test_endpoint_and_optional_hostname_mismatch_are_rejected(self):
        self.assertEqual(self.prepare().returncode, 0)
        for extra in (("--master-port", "29518"), ("--expected-hostname", "deliberately-wrong-host")):
            result = self.launch(*extra)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.base / "runs/dry_run").exists())

    def test_modified_or_old_workspace_is_not_silently_reused(self):
        self.assertEqual(self.prepare().returncode, 0)
        script = self.base / "workspace/scripts/run_pretrain_llama3_full.sh"
        script.write_text(script.read_text() + "\n# changed after preparation\n")
        self.assertNotEqual(self.launch().returncode, 0)
        audit = self.base / "audit/source_manifest.json"
        value = json.loads(audit.read_text())
        value.pop("adapter_schema")
        audit.write_text(json.dumps(value))
        self.assertNotEqual(self.launch().returncode, 0)

    def test_overlapping_source_and_destination_are_rejected_before_copy(self):
        result = self.prepare("--base", str(self.source / "nested-output"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.source / "nested-output").exists())

    def test_unsupported_launch_layout_is_rejected_before_copy(self):
        self.script.write_text(self.script.read_text().replace("MASTER_ADDR=old-example-host\n", ""))
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.base / "workspace").exists())

    def test_existing_workspace_and_invalid_step_window_are_rejected(self):
        self.assertEqual(self.prepare().returncode, 0)
        self.assertNotEqual(self.prepare().returncode, 0)
        self.assertNotEqual(self.launch("--steps", "2", "--exit-interval", "2").returncode, 0)
        self.assertFalse((self.base / "runs/dry_run").exists())


if __name__ == "__main__":
    unittest.main()
