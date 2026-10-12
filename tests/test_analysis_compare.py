import json
import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.compare import compare_runs, inspect_run, _metrics


class AnalysisCompareTests(FixtureCase):
    def test_virtual_pipeline_context_and_optimizer_stage_cannot_be_ignored(self):
        a = self.capture("legacy", [np.ones(1)])
        b = self.capture("null", [(np.ones(1), {"virtual_pipeline": None})])
        c = self.capture("zero", [(np.ones(1), {"virtual_pipeline": 0})])
        self.assertEqual(compare_runs(a, b)["overall"], "pass")
        self.assertEqual(compare_runs(a, c)["overall"], "incomplete")
        d = self.capture("opt1", [(np.ones(1), {"kind": "optimizer_state", "optimizer_type": "Adam", "state_stage": "before"})])
        e = self.capture("opt2", [(np.ones(1), {"kind": "optimizer_state", "optimizer_type": "Adam", "state_stage": "after"})])
        result = compare_runs(d, e)
        self.assertEqual(result["records"][0]["status"], "unaligned")
        self.assertIsNone(result["first_difference"])

    def test_fp32_ulp_and_signed_zero(self):
        a = np.array([-1., -0., 0., 1.], np.float32)
        b = np.array([np.nextafter(a[0], np.float32(-2)), 0., -0., np.nextafter(a[3], np.float32(2))], np.float32)
        result = _metrics(a, b, 1, 1, compute_ulp=True, max_ulp=0)
        self.assertEqual(result["max_ulp_error"], 1)
        self.assertEqual(result["ulp_mismatch_count"], 2)
        self.assertEqual(result["mismatch_count"], 2)
        self.assertEqual(result["tolerance_mismatch_count"], 0)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        result = _metrics(np.array([-tiny]), np.array([tiny]), 1, 1, compute_ulp=True)
        self.assertEqual(result["max_ulp_error"], 2)

    def test_ulp_cannot_hide_nonfinite_and_nonfp32_is_unsupported(self):
        for dtype, fields, expected in ((np.float32, {}, "nonfinite"),
                                         (np.float16, {}, "unsupported"),
                                         (np.float32, {"dtype": "torch.bfloat16"}, "unsupported")):
            name = np.dtype(dtype).name + expected
            a = self.capture(name + "a", [(np.array([np.nan], dtype), fields)])
            b = self.capture(name + "b", [(np.array([np.nan], dtype), fields)])
            self.assertEqual(compare_runs(a, b, max_ulp=999)["records"][0]["status"], expected)

    def test_rules_cli_priority_and_conflict(self):
        a = self.capture("a", [np.array([1.], np.float32)])
        b = self.capture("b", [np.array([1.1], np.float32)])
        rules = {"rules": [{"id": "layer", "module": "model.*", "atol": 1., "rtol": 0.}]}
        result = compare_runs(a, b, rules=rules)
        self.assertEqual(result["overall"], "pass")
        result = compare_runs(a, b, rules=rules, atol=0.)
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["records"][0]["threshold_policy"]["sources"]["atol"], "explicit")
        rules["rules"].append({"module": "*", "atol": .5})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            compare_runs(a, b, rules=rules)
        with self.assertRaises(ValueError):
            compare_runs(a, b, max_ulp=True)

    def test_schema_two_slice_versions_and_missing_fields(self):
        selection = {"original_shape": [4], "selection": [{"start": 0, "stop": 2, "step": 1}]}
        a = self.capture("a", [(np.ones(2), {"tensor_slice": selection})], schema=2)
        b = self.capture("b", [(np.ones(2), {"tensor_slice": selection})], schema=2)
        self.assertEqual(compare_runs(a, b)["overall"], "pass")
        self.assertEqual(inspect_run(a)["events"], 1)
        c = self.capture("c", [(np.ones(2), {"tensor_slice": {**selection,
                    "selection": [{"start": 2, "stop": 4, "step": 1}]}})], schema=2)
        self.assertEqual(compare_runs(a, c)["records"][0]["status"], "unaligned")
        d = self.capture("d", [(np.ones(2), {"tensor_slice": selection, "config_version": 1})], schema=2)
        self.assertEqual(compare_runs(a, d)["overall"], "incomplete")
        path = b / "rank_00000/events.jsonl"
        event = json.loads(path.read_text())
        event.pop("backward_id")
        path.write_text(json.dumps(event) + "\n")
        self.assertEqual(compare_runs(a, b)["overall"], "fail")

    def test_bad_slice_and_schema_one_two_do_not_pass(self):
        a = self.capture("a", [np.ones(2)], schema=1)
        b = self.capture("b", [np.ones(2)], schema=2)
        self.assertEqual(compare_runs(a, b)["overall"], "incomplete")
        c = self.capture("c", [(np.ones(2), {"tensor_slice": {"original_shape": [4],
            "selection": [{"start": 0, "stop": 3, "step": 1}]}})], schema=2)
        self.assertEqual(compare_runs(c, c)["overall"], "fail")

    def test_mapping_preserves_original_and_rejects_structural_mismatch(self):
        a = self.capture("a", [(np.ones(2), {"module": "old"})])
        b = self.capture("b", [(np.ones(2), {"module": "new"})])
        mapping = {"pairs": [{"reference": {"module": "old"}, "candidate": {"module": "new"}}]}
        result = compare_runs(a, b, mapping=mapping)
        self.assertEqual(result["overall"], "pass")
        self.assertEqual(result["records"][0]["candidate"]["module"], "new")
        self.assertEqual(result["records"][0]["mapping"]["candidate"]["original_identity"]["module"], "new")
        c = self.capture("c", [(np.ones(3), {"module": "new"})])
        self.assertEqual(compare_runs(a, c, mapping=mapping)["records"][0]["status"], "shape_mismatch")
        d = self.capture("d", [(np.ones(2), {"module": "old"})])
        self.assertEqual(compare_runs(a, d, mapping=mapping)["overall"], "fail")

    def test_mapping_ambiguous_and_context_rewrite_rejected(self):
        a = self.capture("a", [np.ones(1)])
        bad = {"pairs": [{"reference": {"phase": "forward"}, "candidate": {"phase": "backward"}}]}
        with self.assertRaisesRegex(ValueError, "context"):
            compare_runs(a, a, mapping=bad)
        overlap = {"pairs": [
            {"reference": {"module": "model.linear"}, "candidate": {"module": "x"}},
            {"reference": {"module": "model.linear", "kind": "output"}, "candidate": {"module": "y", "kind": "output"}}]}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            compare_runs(a, a, mapping=overlap)
