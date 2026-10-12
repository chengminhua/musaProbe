import json

import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.compare import compare_runs
from musaprobe.layout_axes import reconstruct_axes


class AxisLayoutTests(FixtureCase):
    def split(self, name, *, overrides=None):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        for rank, part in enumerate((array[:, :2], array[:, 2:])):
            fields = {"kind": "parameter", "parameter_name": f"chunks.{rank}.local.weight", "module": f"chunks.{rank}.local.weight",
                      "virtual_pipeline": rank, **((overrides or {}).get(rank, {}))}
            path = self.capture(name, [(part, fields)], rank=rank, schema=2, world_size=2)
        layout = {"schema_version": 1, "groups": [{"name": "global.weight", "global_shape": [3, 4], "shards": [
            {"rank": 0, "event_id": 0, "offsets": [0, 0]}, {"rank": 1, "event_id": 0, "offsets": [0, 2]}]}]}
        return path, layout, array

    def test_explicit_axis_layout_preserves_declared_scope_and_source_vpp(self):
        capture, layout, expected = self.split("split")
        result = reconstruct_axes(capture, layout, self.root / "out")
        self.assertEqual(result["status"], "complete", result)
        np.testing.assert_array_equal(np.load(self.root / "out/rank_00000/tensor_00000000.npy"), expected)
        event = json.loads((self.root / "out/rank_00000/events.jsonl").read_text())
        self.assertIsNone(event["virtual_pipeline"])
        self.assertEqual([s["virtual_pipeline"] for s in event["axis_reconstruction"]], [0, 1])
        self.assertEqual(event["module"], "global.weight")
        self.assertEqual(compare_runs(self.root / "out", self.root / "out")["overall"], "pass")

    def test_different_explicit_partitionings_compare_after_reconstruction(self):
        a, layout, expected = self.split("a")
        reconstruct_axes(a, layout, self.root / "a_out")
        for rank in range(3):
            b = self.capture("b", [(expected[rank:rank + 1], {"kind": "parameter", "virtual_pipeline": 5})], rank=rank, schema=2, world_size=3)
        other = {"groups": [{"name": "global.weight", "global_shape": [3, 4], "shards": [
            {"rank": rank, "event_id": 0, "offsets": [rank, 0]} for rank in range(3)]}]}
        self.assertEqual(reconstruct_axes(b, other, self.root / "b_out")["status"], "complete")
        self.assertEqual(compare_runs(self.root / "a_out", self.root / "b_out")["overall"], "pass")

    def test_missing_overlap_outside_and_wrong_axes_never_fill_with_zero(self):
        for scenario in ("missing", "overlap", "outside", "dimensions"):
            capture, layout, _ = self.split(scenario)
            group = layout["groups"][0]
            if scenario == "missing":
                group["shards"].pop()
            else:
                group["shards"][1]["offsets"] = {"overlap": [0, 1], "outside": [0, 3], "dimensions": [2]}[scenario]
            result = reconstruct_axes(capture, layout, self.root / (scenario + "_out"))
            self.assertEqual(result["status"], "incomplete", result)
            self.assertFalse(list((self.root / (scenario + "_out/rank_00000")).glob("*.npy")))

    def test_phase_version_backward_stage_and_dtype_must_match(self):
        for index, fields in enumerate(({"phase": "backward"}, {"config_version": 2}, {"backward_id": "other"},
                                        {"gradient_stage": "other"}, {"dtype": "torch.bfloat16"})):
            capture, layout, _ = self.split(str(index), overrides={1: fields})
            result = reconstruct_axes(capture, layout, self.root / (str(index) + "_out"))
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("context", result["records"][0]["reason"])

    def test_reused_missing_event_and_budget_rejected(self):
        for scenario in ("reused", "absent", "budget"):
            capture, layout, _ = self.split(scenario)
            if scenario == "reused":
                layout["groups"][0]["shards"][1]["rank"] = 0
            if scenario == "absent":
                layout["groups"][0]["shards"][1]["event_id"] = 999
            result = reconstruct_axes(capture, layout, self.root / (scenario + "_out"), max_bytes=1 if scenario == "budget" else 1024)
            self.assertEqual(result["status"], "incomplete")

    def test_malformed_declaration_and_output_inside_input_rejected(self):
        capture, layout, _ = self.split("a")
        with self.assertRaises(ValueError):
            reconstruct_axes(capture, layout, capture / "out")
        layout["groups"].append(layout["groups"][0])
        with self.assertRaises(ValueError):
            reconstruct_axes(capture, layout, self.root / "duplicate")
