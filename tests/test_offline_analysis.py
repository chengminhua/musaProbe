import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.analysis import build_trends, compare_configs, suggest_mapping
from musaprobe.graph import analyze_graph


class OfflineAnalysisTests(FixtureCase):
    def test_config_equal_different_unknown_and_types(self):
        fields = ["x.version", "y"]
        self.assertEqual(compare_configs({"x": {"version": "1"}, "y": 0}, {"x": {"version": "1"}, "y": 0}, fields=fields)["overall"], "equal")
        self.assertEqual(compare_configs({"x": 1}, {"x": True}, fields=["x"])["overall"], "different")
        result = compare_configs({"x": None}, {"x": None}, fields=["x", "missing"])
        self.assertEqual(result["overall"], "unknown")
        self.assertFalse(result["records"][1]["reference_present"])
        with self.assertRaises(ValueError):
            compare_configs({}, {}, fields=["x", "x"])

    def test_configs_read_rank_manifests_without_invented_versions(self):
        a = self.capture("a", [np.ones(1)], metadata={"torch_version": "a"})
        b = self.capture("b", [np.ones(1)], metadata={"torch_version": "a"})
        self.assertEqual(compare_configs(a, b, fields=["metadata.torch_version"])["overall"], "equal")
        self.assertEqual(compare_configs(a, b)["overall"], "unknown")
        random_record = next(r for r in compare_configs(a, a)["records"] if r["field"] == "metadata.random_state")
        self.assertEqual(random_record["status"], "unknown")
        with self.assertRaises(ValueError):
            compare_configs(a, b, output=a / "config.json")

    def test_config_nested_rng_unknown_does_not_mask_known_differences(self):
        a = {"random_state": {"torch_cpu": "hash-a", "musa": {"sha256": "hash-m"}, "cuda": None}}
        result = compare_configs(a, a, fields=["random_state"])
        self.assertEqual(result["overall"], "unknown")
        self.assertEqual(result["records"][0]["unknown_paths"], ["random_state.cuda"])
        self.assertEqual(compare_configs(a, a, fields=["random_state.musa.sha256"])["overall"], "equal")
        b = {"random_state": {"torch_cpu": "hash-b", "musa": {"sha256": "hash-m"}, "cuda": None}}
        result = compare_configs(a, b, fields=["random_state"])
        self.assertEqual(result["overall"], "different")
        self.assertEqual(result["records"][0]["different_paths"], ["random_state.torch_cpu"])
        self.assertEqual(result["records"][0]["unknown_paths"], ["random_state.cuda"])
        unknown = {"random_state": {"musa": {"status": "unknown", "reason": "unavailable"}}}
        self.assertEqual(compare_configs(unknown, unknown, fields=["random_state"])["overall"], "unknown")
        self.assertEqual(compare_configs({"x": {"v": [1]}}, {"x": {"v": [True]}}, fields=["x"])["overall"], "different")
        self.assertEqual(compare_configs({"x": {"known": 1}}, {"x": {"known": 1, "missing": 2}}, fields=["x"])["overall"], "unknown")

    def test_suggestions_never_auto_accept_and_keep_ambiguity(self):
        a = self.capture("a", [(np.ones(2), {"module": "old"})])
        b = self.capture("b", [(np.ones(2), {"module": "new"}), (np.ones(2), {"module": "new.other"})])
        result = suggest_mapping(a, b)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["suggestions"][0]["status"], "ambiguous")
        self.assertEqual(len(result["suggestions"][0]["candidates"]), 2)
        self.assertTrue(suggest_mapping(a, b, max_pairs=1)["truncated"])

    def test_suggestions_reject_shape_phase_layout_and_config_context(self):
        a = self.capture("a", [(np.ones(2), {"module": "a"})], schema=2)
        b = self.capture("b", [(np.ones(3), {"module": "b"}), (np.ones(2), {"module": "b", "phase": "backward"}),
                                (np.ones(2), {"module": "b", "config_version": 1})], schema=2)
        self.assertEqual(suggest_mapping(a, b)["suggestions"][0]["status"], "unmatched")
        c = self.capture("c", [(np.ones(2), {"module": "b"})], schema=2, world_size=2)
        self.assertEqual(len(suggest_mapping(a, c)["rejected_ranks"]), 1)

    def test_trends_gaps_nonfinite_and_invalid_statistics(self):
        a = self.capture("trend", [(np.ones(2), {"step": 0, "call_index": 0, "stats": {"numel": 2, "mean": 1}}),
                                  (np.ones(2), {"step": 2, "call_index": 0, "stats": {"numel": 2, "mean": 3, "nan_count": 1}})],
                         config={"steps": [0, 1, 2, 3]})
        result = build_trends(a, output=self.root / "trends")
        self.assertEqual([p["state"] for p in result["series"][0]["points"]], ["available", "missing", "nonfinite", "missing"])
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["series"][0]["points"][1]["metrics"].get("mean"))
        self.check_html(self.root / "trends" / "trends.html")
        b = self.capture("bad_stats", [(np.ones(1), {"stats": {"mean": "bogus"}})])
        self.assertEqual(build_trends(b)["series"][0]["points"][0]["state"], "statistics_invalid")

    def test_trends_duplicate_identity_and_truncated_are_gaps(self):
        a = self.capture("a", [(np.ones(1), {"call_index": 0}), (np.ones(1), {"call_index": 0}),
                                (np.ones(1), {"step": 1, "call_index": 0, "state": "truncated"})])
        states = [p["state"] for p in build_trends(a)["series"][0]["points"]]
        self.assertEqual(states, ["duplicate", "truncated"])

    def graph_capture(self, name, **changes):
        events = [(np.ones(1), {"kind": "api_input", "graph_capture": True, "tensor_id": "external", "producer_event_id": None,
                               "graph_unknown_reason": "external_input", "api_call_index": 0}),
                  (np.ones(1), {"kind": "api_output", "graph_capture": True, "tensor_id": "t1", "input_tensor_ids": ["external"], "api_call_index": 0}),
                  (np.ones(1), {"kind": "api_input", "graph_capture": True, "tensor_id": "t1", "producer_event_id": 1, "api_call_index": 1}),
                  (np.ones(1), {"kind": "api_output", "graph_capture": True, "tensor_id": "t2", "input_tensor_ids": ["t1"], "api_call_index": 1, "stats": {"nan_count": 1}})]
        for index, override in changes.items():
            events[int(index)][1].update(override)
        return self.capture(name, events)

    def test_graph_actual_edges_unknown_roots_and_nonfinite_boundary(self):
        data = analyze_graph(self.graph_capture("g"), output=self.root / "graph")
        self.assertEqual(data["summary"]["edges"], 1)
        self.assertEqual(data["edges"][0]["source"], "r0:e1")
        self.assertEqual(data["edges"][0]["target"], "r0:e3")
        self.assertEqual(data["status"], "partial")
        self.assertEqual(data["observed_nonfinite_boundaries"][0]["node"], "r0:e3")
        self.check_html(self.root / "graph" / "graph.html")

    def test_graph_rejects_fabricated_producer_duplicate_and_unobserved_mutation(self):
        bad = analyze_graph(self.graph_capture("bad", **{"2": {"producer_event_id": 999}}))
        self.assertTrue(bad["errors"])
        dup = analyze_graph(self.graph_capture("dup", **{"3": {"tensor_id": "t1"}}))
        self.assertTrue(dup["errors"])
        self.assertEqual(dup["edges"], [])
        mutation = analyze_graph(self.graph_capture("mut", **{"2": {"producer_event_id": None, "graph_unknown_reason": "unobserved_mutation"}}))
        self.assertEqual(mutation["edges"], [])
        plain = self.capture("plain", [np.ones(1), np.ones(1)])
        self.assertEqual(analyze_graph(plain)["nodes"], [])

    def test_graph_non_tensor_runtime_objects_are_explicit_coverage_gaps(self):
        capture = self.capture("objects", [(np.ones(1), {"kind": "api_input", "state": "unsupported", "dtype": None,
            "graph_capture": True, "reason": "unsupported value type: torch.ScriptObject"})])
        result = analyze_graph(capture)
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["unrepresented_values"]), 1)
        self.assertEqual(result["status"], "partial")

    def test_graph_collective_requires_all_explicit_participants(self):
        def capture(rank, name):
            common = {"operation": "all_reduce", "group_id": "dp", "sequence": 0,
                      "participants": [0, 1], "reduction": "sum", "synchronous": True}
            return self.capture(name, [(np.ones(1), {"kind": "communication_" + stage, "communication": {**common, "stage": stage}})
                                      for stage in ("input", "output")], rank=rank, world_size=2)
        path = capture(0, "comm")
        self.assertEqual(analyze_graph(path)["status"], "incomplete")
        capture(1, "comm")
        result = analyze_graph(path)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["summary"]["edges"], 4)
        self.assertTrue(all(e["evidence"] == "observed_synchronous_collective" for e in result["edges"]))

    def test_offline_import_does_not_require_torch(self):
        code = """import sys
class Deny:
 def find_spec(self, fullname, *args):
  if fullname == 'torch' or fullname.startswith('torch.'):
   raise RuntimeError('torch import forbidden')
sys.meta_path.insert(0,Deny())
import musaprobe.compare, musaprobe.analysis, musaprobe.layout, musaprobe.checkpoint, musaprobe.graph, musaprobe.report
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def check_html(self, path):
        text = path.read_text()
        self.assertNotIn("https://", text)
        self.assertNotIn("innerHTML", text)
        scripts = re.findall(r"<script>(.*?)</script>", text, re.S)
        if shutil.which("node"):
            for script in scripts:
                check = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
                self.assertEqual(check.returncode, 0, check.stderr)
