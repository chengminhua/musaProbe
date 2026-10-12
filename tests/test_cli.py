from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from musaprobe.cli import main


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name, value=1., state="tensor"):
        rank = self.root / name / "rank_00000"
        rank.mkdir(parents=True)
        (rank / "manifest.json").write_text(json.dumps({"schema_version": 1, "rank": 0, "status": "complete"}))
        array = np.array([value], np.float32)
        np.save(rank / "tensor.npy", array, allow_pickle=False)
        event = {"schema_version": 1, "event_id": 0, "step": 0, "microbatch": 0, "phase": "forward",
                 "module": "linear", "call_index": 0, "kind": "output", "tensor_path": "$",
                 "shape": [1], "dtype": "torch.float32", "state": state, "file": "tensor.npy",
                 "stats": {"numel": 1}}
        (rank / "events.jsonl").write_text(json.dumps(event) + "\n")
        return rank.parent

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_inspect_and_report_roundtrip(self):
        reference, candidate = self.capture("ref"), self.capture("cand")
        code, out, err = self.invoke(["inspect", reference])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["events"], 1)
        self.assertFalse(err)
        output = self.root / "compare"
        code, out, err = self.invoke(["compare", reference, candidate, "--output", output])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["overall"], "pass")
        self.assertNotIn("records", json.loads(out))
        report = self.root / "regenerated.html"
        code, out, err = self.invoke(["report", output / "comparison.json", "--output", report])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["report"], str(report))
        self.assertTrue(report.is_file())

    def test_compare_exit_codes_distinguish_mismatch_and_incomplete(self):
        reference = self.capture("ref")
        bad = self.capture("bad", value=10.)
        stats = self.capture("stats", state="statistics")
        self.assertEqual(self.invoke(["compare", reference, bad, "--output", self.root / "bad-report"])[0], 1)
        self.assertEqual(self.invoke(["compare", reference, stats, "--output", self.root / "stats-report"])[0], 2)

    def test_missing_path_and_invalid_tolerance_clean_error(self):
        code, out, err = self.invoke(["inspect", self.root / "missing"])
        self.assertEqual(code, 2)
        self.assertFalse(out)
        self.assertIn("musaProbe:", err)
        self.assertNotIn("Traceback", err)
        ref = self.capture("ref")
        code, out, err = self.invoke(["compare", ref, ref, "--output", self.root / "r", "--atol", "nan"])
        self.assertEqual(code, 2)
        self.assertIn("atol", err)

    def test_replay_unsupported_exit_code(self):
        capture = self.capture("run")
        code, out, err = self.invoke(["replay", capture / "rank_00000", "--event-id", "0"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["status"], "unsupported")
        self.assertFalse(err)

    def test_analysis_cli_does_not_import_torch(self):
        reference, candidate = self.capture("ref"), self.capture("cand")
        args = ["compare", str(reference), str(candidate), "--output", str(self.root / "subprocess-report")]
        script = "import sys; from musaprobe.cli import main; status=main(" + repr(args) + "); assert status==0; assert 'torch' not in sys.modules; assert 'torch_musa' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
