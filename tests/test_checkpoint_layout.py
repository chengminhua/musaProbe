import json
import math
import builtins
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.checkpoint import compare_checkpoints, convert_checkpoint
from musaprobe.compare import compare_runs
from musaprobe.layout import ContiguousTensor, reconstruct_shards


class CheckpointLayoutTests(FixtureCase):
    def test_conversion_without_optional_torch_has_actionable_error(self):
        source = self.root / "weights.pt"
        source.write_bytes(b"unread without PyTorch")
        distributed = self.root / "distributed"
        distributed.mkdir()
        (distributed / ".metadata").write_bytes(b"unread without PyTorch")
        original_import = builtins.__import__
        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch intentionally unavailable")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_torch):
            for path in (source, distributed):
                with self.assertRaisesRegex(ValueError, "requires PyTorch"):
                    convert_checkpoint(path, self.root / "not-written")
        self.assertFalse((self.root / "not-written").exists())

    def bundle(self, name, arrays):
        directory = self.root / name
        directory.mkdir()
        items = []
        for i, (tensor_name, array, fields) in enumerate(arrays):
            array = np.asarray(array)
            item = {"name": tensor_name, "file": f"{i}.npy", "shape": list(array.shape), "dtype": "torch." + array.dtype.name, **fields}
            np.save(directory / f"{i}.npy", array, allow_pickle=False)
            items.append(item)
        (directory / "checkpoint.json").write_text(json.dumps({"format": "musaprobe_checkpoint", "schema_version": 1, "parameters": items}))
        return directory

    def shard_layout(self, start, stop, rank=0, shape=None):
        return {"format": "megatron_contiguous_flat_v1", "original_shape": shape or [2, 2], "flat_range": [start, stop],
                "parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": rank}}

    def shards(self, name, ranges=None, stages=None, tensor_ranks=None, states=None):
        ranges = ranges or [[0, 2], [2, 4]]
        for rank, span in enumerate(ranges):
            fields = {"phase": "backward", "kind": "parameter_grad", "parameter_name": "weight", "module": "weight",
                      "gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [2, 2],
                      "parameter_flat_range": span, "gradient_stage": (stages or ["reduced", "reduced"])[rank],
                      "gradient_parallel_coordinates": {"tensor": (tensor_ranks or [0, 0])[rank], "pipeline": 0, "data": rank}}
            if states:
                fields["state"] = states[rank]
            capture = self.capture(name, [(np.arange(*span, dtype=np.float32), fields)], rank=rank, world_size=len(ranges))
        return capture

    def test_checkpoint_full_integer_exact_nonfinite_missing(self):
        a = self.bundle("a", [("w", np.array([2**63 - 1], dtype=np.int64), {})])
        b = self.bundle("b", [("w", np.array([2**63 - 2], dtype=np.int64), {})])
        result = compare_checkpoints(a, b, atol=1e10, output=self.root / "report")
        self.assertEqual(result["records"][0]["metrics"]["max_abs_error"], 1)
        self.assertEqual(result["overall"], "fail")
        self.assertTrue((self.root / "report" / "report.html").exists())
        c = self.bundle("c", [("other", np.ones(1), {})])
        self.assertEqual(compare_checkpoints(a, c)["overall"], "incomplete")
        d = self.bundle("d", [("w", np.array([np.nan]), {})])
        self.assertEqual(compare_checkpoints(d, d)["records"][0]["status"], "nonfinite")

    def test_checkpoint_shards_compare_to_full_without_full_allocation(self):
        array = np.arange(4, dtype=np.float32).reshape(2, 2)
        full = self.bundle("full", [("w", array, {})])
        shards = self.bundle("shards", [("w", array.ravel()[:2], {"layout": self.shard_layout(0, 2)}),
                                         ("w", array.ravel()[2:], {"layout": self.shard_layout(2, 4, 1)})])
        result = compare_checkpoints(full, shards, max_ulp=0)
        self.assertEqual(result["overall"], "pass", result)
        self.assertEqual(result["records"][0]["metrics"]["numel"], 4)

    def test_checkpoint_rejects_gap_overlap_missing_layout_duplicate_and_traversal(self):
        full = self.bundle("full", [("w", np.arange(4, dtype=np.float32).reshape(2, 2), {})])
        for label, specs in [("gap", [(0, 1, 0), (2, 4, 1)]), ("overlap", [(0, 3, 0), (2, 4, 1)]), ("owner", [(0, 2, 0), (2, 4, 0)])]:
            path = self.bundle(label, [("w", np.arange(start, stop, dtype=np.float32), {"layout": self.shard_layout(start, stop, rank)})
                                       for start, stop, rank in specs])
            self.assertEqual(compare_checkpoints(full, path)["records"][0]["status"], "error")
        unknown = self.bundle("unknown", [("w", np.arange(4, dtype=np.float32), {"sharded": True})])
        self.assertEqual(compare_checkpoints(full, unknown)["overall"], "fail")
        duplicate = self.bundle("duplicate", [("w", np.ones(1), {}), ("w", np.ones(1), {})])
        self.assertEqual(compare_checkpoints(duplicate, duplicate)["overall"], "fail")
        escape = self.bundle("escape", [("w", np.ones(1), {"file": "../outside.npy"})])
        self.assertIn("unsafe", compare_checkpoints(escape, escape)["records"][0]["reason"])

    def test_checkpoint_conversion_is_explicit_weights_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch conversion is optional")
        source = self.root / "weights.pt"
        torch.save({"model": {"w": torch.tensor([1., 2.]), "bf": torch.tensor([1.], dtype=torch.bfloat16)}}, source)
        result = convert_checkpoint(source, self.root / "converted")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["parameters"][1]["dtype"], "torch.bfloat16")
        self.assertEqual(compare_checkpoints(self.root / "converted", self.root / "converted")["overall"], "pass")
        with self.assertRaises(ValueError):
            compare_checkpoints(source, source)
        torch.save({"model": {"w": torch.tensor([1.])}, "state_dict": {"w": torch.tensor([1.])}}, self.root / "ambiguous.pt")
        with self.assertRaises(ValueError):
            convert_checkpoint(self.root / "ambiguous.pt", self.root / "ambiguous")
        import argparse
        torch.save({"args": argparse.Namespace(hidden_size=8), "model": {"w": torch.ones(1)}}, self.root / "legacy.pt")
        with self.assertRaisesRegex(ValueError, "weights_only=True"):
            convert_checkpoint(self.root / "legacy.pt", self.root / "legacy_out", key="model")

    def test_reconstruct_shards_complete_and_output_self_comparison(self):
        result = reconstruct_shards(self.shards("shards"), self.root / "assembled")
        self.assertEqual(result["status"], "complete", result)
        array = np.load(self.root / "assembled" / "rank_00000" / "tensor_00000000.npy")
        np.testing.assert_array_equal(array, np.arange(4, dtype=np.float32).reshape(2, 2))
        compared = compare_runs(self.root / "assembled", self.root / "assembled")
        self.assertEqual(compared["overall"], "pass", compared)
        with self.assertRaises(ValueError):
            reconstruct_shards(self.root / "assembled", self.root / "assembled")

    def test_reconstruction_refuses_missing_shards_overlap_stage_and_budget(self):
        for label, ranges in [("missing", [[0, 2]]), ("overlap", [[0, 3], [2, 4]]), ("gap", [[0, 1], [2, 4]])]:
            result = reconstruct_shards(self.shards(label, ranges=ranges), self.root / (label + "_out"))
            self.assertEqual(result["status"], "incomplete", result)
        stage = reconstruct_shards(self.shards("stage", stages=["pre", "post"]), self.root / "stage_out")
        self.assertEqual(stage["status"], "incomplete")
        truncated = reconstruct_shards(self.shards("truncated", states=["tensor", "truncated"]), self.root / "trunc_out")
        self.assertEqual(truncated["status"], "incomplete")
        budget = reconstruct_shards(self.shards("budget"), self.root / "budget_out", max_bytes=1)
        self.assertEqual(budget["status"], "incomplete")
        self.assertEqual(budget["bytes"], 0)

    def test_reconstruction_does_not_merge_distinct_tp_groups(self):
        result = reconstruct_shards(self.shards("tp", tensor_ranks=[0, 1]), self.root / "tp_out")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 2)

    def test_reconstruction_complete_tp_groups_remain_distinct_ranks(self):
        for rank in range(4):
            dp, tp = rank % 2, rank // 2
            span = [dp * 2, dp * 2 + 2]
            capture = self.capture("multi_tp", [(np.arange(*span, dtype=np.float32) + tp * 10,
                {"phase": "backward", "kind": "parameter_grad", "parameter_name": "w", "module": "w",
                 "gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [4],
                 "parameter_flat_range": span, "gradient_stage": "reduced",
                 "gradient_parallel_coordinates": {"tensor": tp, "pipeline": 0, "data": dp}})], rank=rank, world_size=4)
        result = reconstruct_shards(capture, self.root / "multi_tp_out")
        self.assertEqual(result["status"], "complete", result)
        self.assertTrue((self.root / "multi_tp_out/rank_00001/manifest.json").exists())
        self.assertEqual(compare_runs(self.root / "multi_tp_out", self.root / "multi_tp_out")["overall"], "pass")

    def test_contiguous_tensor_empty_and_chunk_reader(self):
        tensor = ContiguousTensor([4], [(0, 2, np.array([1, 2])), (2, 4, np.array([3, 4]))])
        np.testing.assert_array_equal(tensor.flat[1:3], [2, 3])
        empty = ContiguousTensor([0, 2], [(0, 0, np.empty((0, 2), dtype=np.float32))])
        self.assertEqual(empty.flat[:].shape, (0,))
        with self.assertRaises(ValueError):
            ContiguousTensor([4], [(0, 2, np.array([1, 2])), (2, 4, np.array([3., 4.]))])
