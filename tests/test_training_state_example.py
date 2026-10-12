"""CPU checks for the executable's artifact-based acceptance, not GPU claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


_PATH = Path(__file__).resolve().parents[1] / "examples/training_state_check.py"
_SPEC = importlib.util.spec_from_file_location("training_state_example", _PATH)
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)


class TrainingStateExampleTests(unittest.TestCase):
    def test_real_adam_state_payloads_and_three_step_trends(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _EXAMPLE.optimizer_check(Path(directory), "cpu", torch.float32)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["events"], 18)
            self.assertEqual(result["trend_series"], 6)
            self.assertTrue(all(result["assertions"].values()))

    def test_false_adapter_success_cannot_hide_missing_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                _EXAMPLE, "capture_optimizer_state", return_value={"status": "pass", "captured": [{}] * 6}):
            with self.assertRaisesRegex(AssertionError, "all_18_state_payloads_complete"):
                _EXAMPLE.optimizer_check(Path(directory), "cpu")

    def test_mcore_requires_actual_source_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "actual Megatron source tree"):
                _EXAMPLE.mcore_checkpoint_check(Path(directory), "cpu", 0, 1, Path(directory))

    def test_mcore_failure_cleanup_preserves_exception_and_joins_only_owned_writer(self):
        # Lifecycle-only fixture: this does not serialize or represent a MCore checkpoint.
        modules = {name: ModuleType(name) for name in (
            "megatron", "megatron.core", "megatron.core.dist_checkpointing",
            "megatron.core.dist_checkpointing.strategies")}
        pending = []
        queue = SimpleNamespace(async_calls=pending, get_num_unfinalized_calls=lambda: len(pending))
        filesystem = SimpleNamespace(_results_queue=None)
        modules["megatron.core.dist_checkpointing.strategies"].base = SimpleNamespace(async_calls=queue)
        modules["megatron.core.dist_checkpointing.strategies"].filesystem_async = filesystem
        process = Mock(pid=101, exitcode=0)
        process.is_alive.return_value = False
        manager = Mock()
        def fail_save(*args, **kwargs):
            pending.append(SimpleNamespace(async_caller=SimpleNamespace(process=process)))
            filesystem._results_queue = SimpleNamespace(_manager=manager)
            raise RuntimeError("controlled original synchronization failure")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "controlled original synchronization failure"):
                _EXAMPLE._save_mcore_with_cleanup(SimpleNamespace(save=fail_save), {}, root / "checkpoint", root, 0)
            audit = json.loads((root / "mcore_cleanup_rank_0.json").read_text())
            self.assertFalse(audit["save_returned"])
            self.assertTrue(audit["cleanup_complete"])
            process.join.assert_called_once_with(timeout=10)
            manager.shutdown.assert_called_once_with()
            self.assertIsNone(filesystem._results_queue)
            process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
