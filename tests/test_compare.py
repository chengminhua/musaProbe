from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from musaprobe.compare import compare_runs, inspect_run


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name, values, *, rank=0, status="complete", metadata=None, world_size=1, config=None):
        directory = self.root / name / f"rank_{rank:05d}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "rank": rank,
                "status": status, "metadata": metadata or {}, "world_size": world_size, "config": config or {}}), encoding="utf-8")
        events = []
        for i, value in enumerate(values):
            overrides = {}
            if isinstance(value, tuple):
                value, overrides = value
            array = np.asarray(value)
            event = {"schema_version": 1, "event_id": i, "step": 0, "microbatch": 0,
                     "phase": "forward", "module": "model.linear", "call_index": i,
                     "kind": "output", "tensor_path": "$", "op": None,
                     "shape": list(array.shape), "dtype": "torch." + array.dtype.name,
                     "state": "tensor", "file": f"tensor_{i}.npy",
                     "stats": {"numel": array.size}, "stack": [{"file": "model.py", "line": 9, "function": "forward"}]}
            event.update(overrides)
            if event["state"] == "tensor":
                np.save(directory / f"tensor_{i}.npy", array, allow_pickle=False)
            events.append(event)
        (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return directory.parent

    def compare_values(self, a, b, **kwargs):
        return compare_runs(self.capture("ref", [a]), self.capture("cand", [b]), **kwargs)

    def test_float_pass_mismatch_and_first_observation(self):
        ref = self.capture("ref", [np.array([1., 2.], np.float32)] * 3)
        cand = self.capture("cand", [np.array([1., 2.], np.float32), np.array([1., 4.], np.float32), np.array([8., 9.], np.float32)])
        result = compare_runs(ref, cand)
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["summary"]["statuses"], {"pass": 1, "mismatch": 2})
        self.assertEqual(result["first_difference"]["identity"]["call_index"], 1)
        metrics = result["records"][1]["metrics"]
        self.assertEqual(metrics["max_abs_error"], 2)
        self.assertEqual(metrics["mismatch_ratio"], .5)
        self.assertAlmostEqual(metrics["relative_l2"], 2 / np.sqrt(5))

    def test_integer_comparison_does_not_lose_large_values(self):
        result = self.compare_values(np.array([2**62], np.int64), np.array([2**62 + 1], np.int64), atol=100, rtol=100)
        self.assertEqual(result["records"][0]["status"], "mismatch")
        self.assertEqual(result["records"][0]["metrics"]["mismatch_count"], 1)
        self.assertTrue(result["records"][0]["metrics"]["exact_comparison"])

    def test_bool_is_exact(self):
        result = self.compare_values(np.array([True, False]), np.array([True, True]), atol=10)
        self.assertEqual(result["records"][0]["metrics"]["max_abs_error"], 1)
        self.assertEqual(result["overall"], "fail")

    def test_empty_and_scalar(self):
        ref = self.capture("ref", [np.array([], dtype=np.float32), np.array(2., dtype=np.float32)])
        cand = self.capture("cand", [np.array([], dtype=np.float32), np.array(2., dtype=np.float32)])
        result = compare_runs(ref, cand)
        self.assertEqual(result["overall"], "pass")
        self.assertEqual(result["records"][0]["metrics"]["numel"], 0)
        self.assertEqual(result["records"][0]["metrics"]["mismatch_ratio"], 0)

    def test_nonfinite_is_failure_even_if_equal(self):
        array = np.array([np.nan, np.inf, -np.inf, 1.], np.float32)
        result = self.compare_values(array, array)
        self.assertEqual(result["records"][0]["status"], "nonfinite")
        self.assertEqual(result["records"][0]["metrics"]["nonfinite_candidate"], 3)
        self.assertEqual(result["records"][0]["metrics"]["finite_pairs"], 1)
        json.dumps(result, allow_nan=False)

    def test_bfloat16_values_stored_in_float32(self):
        result = self.compare_values((np.array([1.], np.float32), {"dtype": "torch.bfloat16"}),
                                     (np.array([1.01], np.float32), {"dtype": "torch.bfloat16"}))
        self.assertEqual(result["overall"], "pass")
        self.assertEqual(result["records"][0]["metrics"]["atol"], 1e-2)

    def test_shape_and_dtype_mismatch(self):
        ref = self.capture("ref", [np.ones((2,), np.float32), np.ones((2,), np.float32)])
        cand = self.capture("cand", [np.ones((1, 2), np.float32), np.ones((2,), np.float16)])
        self.assertEqual([r["status"] for r in compare_runs(ref, cand)["records"]], ["shape_mismatch", "dtype_mismatch"])

    def test_statistics_truncated_unsupported_never_pass(self):
        states = ["statistics", "truncated", "unsupported", "error"]
        values = [(np.ones(1, np.float32), {"state": state, "file": None}) for state in states]
        result = compare_runs(self.capture("ref", values), self.capture("cand", values))
        self.assertEqual([r["status"] for r in result["records"]], ["statistics_only", "truncated", "unsupported", "error"])
        self.assertEqual(result["overall"], "fail")

    def test_statistics_nonfinite_detection(self):
        value = (np.ones(1, np.float32), {"state": "statistics", "file": None, "stats": {"nan_count": 1}})
        result = self.compare_values(value, value)
        self.assertEqual(result["records"][0]["status"], "nonfinite")

    def test_statistics_only_is_incomplete_even_when_identical(self):
        value = (np.ones(1, np.float32), {"state": "statistics", "file": None})
        result = self.compare_values(value, value)
        self.assertEqual(result["overall"], "incomplete")
        self.assertIsNone(result["first_difference"])

    def test_budget_notice_without_dtype(self):
        value = (np.array(0), {"state": "truncated", "dtype": None, "kind": "notice", "reason": "max_events reached"})
        result = self.compare_values(value, value)
        self.assertEqual(result["overall"], "incomplete")
        self.assertEqual(result["records"][0]["status"], "truncated")

    def test_missing_events_and_ranks(self):
        ref = self.capture("ref", [np.ones(1), (np.ones(1), {"module": "reference_only"})])
        cand = self.capture("cand", [np.ones(1), (np.ones(1), {"module": "candidate_only"})])
        self.capture("ref", [np.ones(1)], rank=1)
        result = compare_runs(ref, cand)
        self.assertEqual(result["overall"], "incomplete")
        self.assertEqual(result["summary"]["statuses"]["missing_candidate"], 2)
        self.assertEqual(result["summary"]["statuses"]["missing_reference"], 1)

    def test_completed_filtered_ranks_do_not_make_full_run_incomplete(self):
        for name in ("ref", "cand"):
            for rank in range(8):
                self.capture(name, [np.ones(1, np.float32)] if rank in (0, 4) else [], rank=rank,
                             world_size=8, config={"ranks": [0, 4]})
        result = compare_runs(self.root / "ref", self.root / "cand")
        self.assertEqual(result["overall"], "pass")
        self.assertEqual(result["summary"]["statuses"], {"pass": 2})
        self.assertEqual({record["rank"] for record in result["records"]}, {0, 4})
        for side in ("reference", "candidate"):
            coverage = result["coverage"][side]
            self.assertEqual(coverage["recorded_ranks"], [0, 4])
            for rank in coverage["ranks"]:
                self.assertEqual(rank["configured_rank_filter"], [0, 4])
                self.assertEqual(rank["excluded_by_rank_filter"], rank["rank"] not in (0, 4))

    def test_duplicate_identity_is_reported_without_overwrite(self):
        values = [(np.ones(1), {"call_index": 0})] * 2
        result = compare_runs(self.capture("ref", values), self.capture("cand", [np.ones(1)]))
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["records"][0]["status"], "unaligned")
        self.assertEqual(result["records"][0]["duplicate_reference_count"], 2)
        self.assertIsNone(result["first_difference"])
        self.assertEqual(inspect_run(self.root / "ref")["duplicate_identities"][0]["count"], 2)

    def test_recompute_and_api_call_indexes_are_part_of_identity(self):
        ref = self.capture("ref", [(np.ones(1), {"recompute": False}), (np.ones(1), {"op": "aten.add.Tensor", "api_call_index": 0})])
        cand = self.capture("cand", [(np.ones(1), {"recompute": True}), (np.ones(1), {"op": "aten.add.Tensor", "api_call_index": 1})])
        result = compare_runs(ref, cand)
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(result["overall"], "incomplete")

    def test_semantic_alignment_ignores_event_id_and_device(self):
        result = self.compare_values((np.ones(1), {"event_id": 9, "device": "cpu"}),
                                     (np.ones(1), {"event_id": 90, "device": "musa:0"}))
        self.assertEqual(result["overall"], "pass")

    def test_different_parallel_layout_not_compared(self):
        ref = self.capture("ref", [np.ones(1)], metadata={"parallelism": {"tp": 1, "pp": 2}})
        cand = self.capture("cand", [np.ones(1)], metadata={"parallelism": {"tp": 2, "pp": 1}})
        result = compare_runs(ref, cand)
        self.assertEqual(result["records"][0]["status"], "unaligned")
        self.assertEqual(result["summary"]["verified_tensors"], 0)
        self.assertIsNone(result["first_difference"])

    @staticmethod
    def shard_layout():
        return {"kind": "parameter_grad", "parameter_name": "weight",
                "gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [4],
                "parameter_flat_range": [0, 2], "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip",
                "gradient_parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 0}}

    def test_equal_flat_tensor_with_different_parameter_shard_is_unaligned(self):
        ref = (np.array([1., 2.], np.float32), self.shard_layout())
        cand = (np.array([1., 2.], np.float32), {**self.shard_layout(), "parameter_flat_range": [2, 4]})
        result = self.compare_values(ref, cand)
        record = result["records"][0]
        self.assertEqual(record["status"], "unaligned")
        self.assertIn("parameter_flat_range differs", record["reason"])
        self.assertEqual(record["metrics"], {})
        self.assertEqual(record["gradient_layout"]["candidate"]["parameter_flat_range"], [2, 4])
        self.assertIsNone(result["first_difference"])

    def test_gradient_stage_coordinates_and_original_shape_must_match(self):
        alternatives = {"gradient_stage": "post_unscale_post_clip",
                        "gradient_parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 1},
                        "original_parameter_shape": [2, 2]}
        for number, (field, value) in enumerate(alternatives.items()):
            with self.subTest(field=field):
                ref = self.capture(f"ref{number}", [(np.ones(2, np.float32), self.shard_layout())])
                cand = self.capture(f"cand{number}", [(np.ones(2, np.float32), {**self.shard_layout(), field: value})])
                result = compare_runs(ref, cand)
                self.assertEqual(result["records"][0]["status"], "unaligned")
                self.assertIn(field + " differs", result["records"][0]["reason"])

    def test_old_and_new_gradient_metadata_cannot_silently_align(self):
        legacy = {"kind": "parameter_grad", "parameter_name": "weight"}
        for number, (left, right) in enumerate(((legacy, self.shard_layout()), (self.shard_layout(), legacy))):
            ref = self.capture(f"ref{number}", [(np.ones(2, np.float32), left)])
            cand = self.capture(f"cand{number}", [(np.ones(2, np.float32), right)])
            result = compare_runs(ref, cand)
            self.assertEqual(result["records"][0]["status"], "unaligned")
            self.assertIn("present on only one side", result["records"][0]["reason"])

    def test_owned_shard_requires_its_layout_even_if_both_sides_omit_it(self):
        for number, field in enumerate(("original_parameter_shape", "parameter_flat_range", "gradient_stage", "gradient_parallel_coordinates")):
            layout = self.shard_layout()
            layout.pop(field)
            ref = self.capture(f"ref{number}", [(np.ones(2, np.float32), layout)])
            cand = self.capture(f"cand{number}", [(np.ones(2, np.float32), layout)])
            result = compare_runs(ref, cand)
            self.assertEqual(result["records"][0]["status"], "unaligned")
            self.assertIn("optimizer shard lacks " + field, result["records"][0]["reason"])

    def test_same_owned_shard_uses_unchanged_numeric_rules(self):
        ref = self.capture("ref", [(np.ones(2, np.float32), self.shard_layout())] * 2)
        cand = self.capture("cand", [(np.ones(2, np.float32), self.shard_layout()),
                                     (np.array([1., 1.01], np.float32), self.shard_layout())])
        result = compare_runs(ref, cand)
        self.assertEqual([record["status"] for record in result["records"]], ["pass", "mismatch"])
        self.assertEqual(result["records"][1]["metrics"]["atol"], 1e-5)
        self.assertEqual(result["first_difference"]["identity"]["call_index"], 1)

    def test_full_parameter_and_legacy_gradients_keep_supported_comparison(self):
        variants = [{"kind": "parameter_grad", "parameter_name": "weight"},
                    {"kind": "parameter_grad", "parameter_name": "weight", "gradient_scope": "full_parameter",
                     "original_parameter_shape": [2]}]
        for number, layout in enumerate(variants):
            ref = self.capture(f"ref{number}", [(np.ones(2, np.float32), layout)])
            cand = self.capture(f"cand{number}", [(np.ones(2, np.float32), layout)])
            self.assertEqual(compare_runs(ref, cand)["overall"], "pass")

    def test_not_owned_gradient_is_unsupported_and_never_a_pass(self):
        layout = {"kind": "parameter_grad", "parameter_name": "weight", "gradient_scope": "not_owned_by_rank",
                  "original_parameter_shape": [4], "parameter_flat_range": None,
                  "state": "unsupported", "dtype": None, "file": None}
        result = self.compare_values((np.array([]), layout), (np.array([]), layout))
        self.assertEqual(result["overall"], "incomplete")
        self.assertEqual(result["records"][0]["status"], "unsupported")
        self.assertEqual(result["summary"]["verified_tensors"], 0)

    def test_owned_and_not_owned_gradient_cannot_align(self):
        nonowned = {**self.shard_layout(), "gradient_scope": "not_owned_by_rank", "parameter_flat_range": None,
                    "state": "unsupported", "file": None}
        result = self.compare_values((np.ones(2, np.float32), self.shard_layout()), (np.ones(2, np.float32), nonowned))
        self.assertEqual(result["records"][0]["status"], "unaligned")

    def test_malformed_json_and_interrupted_tail_preserve_other_records(self):
        ref = self.capture("ref", [np.ones(1)])
        cand = self.capture("cand", [np.ones(1)], status="running")
        with (cand / "rank_00000/events.jsonl").open("a") as stream:
            stream.write('{"incomplete":')
        result = compare_runs(ref, cand)
        self.assertEqual(result["summary"]["statuses"]["pass"], 1)
        self.assertEqual(result["summary"]["statuses"]["error"], 1)
        self.assertEqual(result["overall"], "fail")
        self.assertIsNone(result["first_difference"])

    def test_capture_error_does_not_hide_first_numeric_difference(self):
        ref = self.capture("ref", [np.array([1.], np.float32)])
        cand = self.capture("cand", [np.array([2.], np.float32)])
        with (cand / "rank_00000/events.jsonl").open("a") as stream:
            stream.write('{"unfinished":')
        result = compare_runs(ref, cand)
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["records"][0]["status"], "error")
        self.assertEqual(result["first_difference"]["status"], "mismatch")

    def test_running_manifest_cannot_pass(self):
        ref = self.capture("ref", [np.ones(1)])
        cand = self.capture("cand", [np.ones(1)], status="running")
        self.assertEqual(compare_runs(ref, cand)["overall"], "incomplete")

    def test_complete_manifest_detects_lost_event_tail(self):
        ref = self.capture("ref", [np.ones(1)])
        cand = self.capture("cand", [np.ones(1)])
        path = cand / "rank_00000/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["events_count"] = 2
        path.write_text(json.dumps(manifest))
        result = compare_runs(ref, cand)
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["summary"]["statuses"]["pass"], 1)
        self.assertEqual(result["summary"]["statuses"]["error"], 1)

    def test_noninteger_event_id_is_reported_as_error(self):
        result = self.compare_values((np.ones(1), {"event_id": "bad"}), np.ones(1))
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["summary"]["statuses"]["error"], 1)

    def test_unsafe_paths_are_rejected(self):
        for index, path in enumerate(("../outside.npy", "/tmp/outside.npy")):
            ref = self.capture(f"ref{index}", [(np.ones(1), {"file": path})])
            cand = self.capture(f"cand{index}", [np.ones(1)])
            result = compare_runs(ref, cand)
            self.assertEqual(result["records"][0]["status"], "error")
            self.assertIn("unsafe", result["records"][0]["reason"])

    def test_symlink_escape_is_rejected(self):
        ref = self.capture("ref", [(np.ones(1), {"file": "link.npy"})])
        cand = self.capture("cand", [np.ones(1)])
        (ref / "rank_00000/link.npy").symlink_to(cand / "rank_00000/tensor_0.npy")
        self.assertIn("symlink", compare_runs(ref, cand)["records"][0]["reason"])

    def test_payload_metadata_and_object_files_are_rejected(self):
        ref = self.capture("ref", [(np.ones(1), {"shape": [2]})])
        cand = self.capture("cand", [np.ones(2)])
        self.assertEqual(compare_runs(ref, cand)["records"][0]["status"], "error")
        np.save(ref / "rank_00000/tensor_0.npy", np.array([object(), object()]))
        self.assertEqual(compare_runs(ref, cand)["records"][0]["status"], "error")

    def test_chunked_fortran_and_zero_reference(self):
        ref = np.arange(12, dtype=np.float64).reshape(3, 4)
        cand = np.asfortranarray(ref.copy())
        cand[0, 0] = 1
        with mock.patch("musaprobe.compare._CHUNK_ELEMENTS", 3):
            result = self.compare_values(ref, cand)
        metrics = result["records"][0]["metrics"]
        self.assertEqual(metrics["mismatch_count"], 1)
        self.assertTrue(metrics["relative_error_unbounded"])
        self.assertIsNone(metrics["max_relative_error"])
        self.assertAlmostEqual(metrics["cosine_similarity"], np.dot(ref.flat, cand.flat)/(np.linalg.norm(ref)*np.linalg.norm(cand)))

    def test_inspection_metadata_summary(self):
        path = self.capture("run", [(np.ones(1), {"state": "statistics", "stats": {"nan_count": 1}})])
        result = inspect_run(path / "rank_00000")
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["nonfinite"][0]["nan_count"], 1)
        self.assertEqual(result["ranks"][0]["steps"], [0])

    def test_outputs_json_csv_html_and_no_capture_mutation(self):
        ref, cand = self.capture("ref", [np.ones(1)]), self.capture("cand", [np.ones(1)])
        before = sorted(str(p.relative_to(ref)) for p in ref.rglob("*"))
        result = compare_runs(ref, cand, output=self.root / "reports")
        self.assertEqual(json.loads((self.root / "reports/comparison.json").read_text()), result)
        with (self.root / "reports/comparison.csv").open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["status"], "pass")
        self.assertTrue((self.root / "reports/report.html").is_file())
        self.assertEqual(before, sorted(str(p.relative_to(ref)) for p in ref.rglob("*")))
        with self.assertRaises(ValueError):
            compare_runs(ref, cand, output=ref / "reports")

    def test_csv_formula_injection_escaped(self):
        ref = self.capture("ref", [(np.ones(1), {"module": "=malicious()"})])
        cand = self.capture("cand", [(np.ones(1), {"module": "=malicious()"})])
        compare_runs(ref, cand, output=self.root / "reports")
        with (self.root / "reports/comparison.csv").open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["module"], "'=malicious()")

    def test_invalid_tolerances_and_empty_capture(self):
        ref, cand = self.capture("ref", []), self.capture("cand", [])
        self.assertEqual(compare_runs(ref, cand)["overall"], "incomplete")
        for value in (-1, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                compare_runs(ref, cand, atol=value)

    def test_offline_import_does_not_import_torch(self):
        command = "import sys; import musaprobe.compare; import musaprobe.report; assert 'torch' not in sys.modules; assert 'torch_musa' not in sys.modules"
        subprocess.run([sys.executable, "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
