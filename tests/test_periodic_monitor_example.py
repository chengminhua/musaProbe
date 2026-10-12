"""CPU contract regression; MUSA acceptance requires the explicit example run."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


_PATH = Path(__file__).resolve().parents[1] / "examples/periodic_monitor_check.py"
_SPEC = importlib.util.spec_from_file_location("periodic_monitor_example", _PATH)
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)


class PeriodicMonitorExampleTests(unittest.TestCase):
    def test_six_real_steps_with_sampling_gaps_and_exact_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _EXAMPLE.exercise(Path(directory), "cpu", torch.float32)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["captured_steps"], [1, 3])
            self.assertEqual(result["events"], 24)
            self.assertEqual(result["trend_status"], "incomplete")
            self.assertTrue(all(result["assertions"].values()))

    def test_absent_parameter_payloads_cannot_be_hidden_by_equal_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                _EXAMPLE.PrecisionDebugger, "capture_parameter_grads", return_value=None):
            with self.assertRaisesRegex(AssertionError, "selected_steps_have_all_required_kinds"):
                _EXAMPLE.exercise(Path(directory), "cpu", torch.float32)


if __name__ == "__main__":
    unittest.main()
