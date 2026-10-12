"""CPU acceptance of the target-device example and its no-false-pass checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


_PATH = Path(__file__).resolve().parents[1] / "examples/checkpoint_lifecycle_check.py"
_SPEC = importlib.util.spec_from_file_location("checkpoint_lifecycle_example", _PATH)
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)


class CheckpointLifecycleExampleTests(unittest.TestCase):
    def test_non_reentrant_shared_module_two_backwards_float32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                result = _EXAMPLE.exercise(Path(directory), "cpu", dtype)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["checkpoint_entries_per_two_backwards"], 3)
                self.assertTrue(all(result["assertions"].values()))

    def test_reentrant_real_checkpoint_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _EXAMPLE.exercise(Path(directory), "cpu", torch.float32, reentrant=True)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["use_reentrant"])
            self.assertTrue(all(result["assertions"].values()))

    def test_missing_parameter_capture_does_not_pass_even_if_gradients_match(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                _EXAMPLE.PrecisionDebugger, "capture_parameter_grads", return_value=None):
            with self.assertRaisesRegex(AssertionError, "all_named_parameter_gradient_files"):
                _EXAMPLE.exercise(Path(directory), "cpu", torch.float32)


if __name__ == "__main__":
    unittest.main()
