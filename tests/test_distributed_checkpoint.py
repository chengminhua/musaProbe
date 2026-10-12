import io
import json
import os
import pickle
import shutil
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.checkpoint import convert_checkpoint, compare_checkpoints
from musaprobe.distributed_checkpoint import convert_torch_dist, _metadata, _rectangles


class DistributedCheckpointTests(FixtureCase):
    def setUp(self):
        super().setUp()
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch checkpoint converter unavailable")
        self.torch = torch

    def actual_checkpoint(self, name):
        import torch.distributed.checkpoint as dcp
        directory = self.root / name
        self.weights = {"model.weight": self.torch.arange(12, dtype=self.torch.float32).reshape(3, 4),
                        "model.bfloat": self.torch.ones(2, dtype=self.torch.bfloat16)}
        dcp.save(self.weights, checkpoint_id=str(directory))
        return directory

    def test_actual_pytorch_distributed_serialization_round_trip(self):
        source = self.actual_checkpoint("dcp")
        result = convert_checkpoint(source, self.root / "converted")
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(len(result["parameters"]), 2)
        for item in result["parameters"]:
            np.testing.assert_array_equal(np.load(self.root / "converted" / item["file"]), self.weights[item["name"]].float().numpy())
        self.assertEqual(compare_checkpoints(self.root / "converted", self.root / "converted")["overall"], "pass")

    def test_nd_reformulated_missing_data_and_budget_are_incomplete(self):
        source = self.actual_checkpoint("nd")
        metadata = _metadata(source / ".metadata")
        metadata.mcore_data = {"model.weight": {"nd_reformulated_orig_global_shape": [12]}}
        (source / ".metadata").write_bytes(pickle.dumps(metadata))
        result = convert_checkpoint(source, self.root / "nd_out")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("N-D", result["unsupported"][0]["reason"])
        missing = self.actual_checkpoint("missing")
        next(missing.glob("*.distcp")).unlink()
        self.assertEqual(convert_checkpoint(missing, self.root / "missing_out")["status"], "incomplete")
        small = self.actual_checkpoint("budget")
        self.assertEqual(convert_torch_dist(small, self.root / "budget_out", max_chunk_bytes=1)["status"], "incomplete")

    def test_metadata_rejects_unknown_pickle_class_without_execution(self):
        marker = self.root / "must_not_exist"
        class Payload:
            def __reduce__(self):
                return os.system, ("touch " + str(marker),)
        path = self.root / ".metadata"
        path.write_bytes(pickle.dumps(Payload()))
        with self.assertRaisesRegex(ValueError, "metadata class"):
            _metadata(path)
        self.assertFalse(marker.exists())

    def test_rectangular_coverage_rejects_holes_overlap_and_duplicates(self):
        from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
        chunk = lambda a, b: ChunkStorageMetadata(self.torch.Size(a), self.torch.Size(b))
        result = _rectangles((2, 4), [chunk([0, 0], [2, 2]), chunk([0, 2], [2, 2])])
        self.assertEqual(len(result), 2)
        for chunks in ([chunk([0, 0], [2, 2])], [chunk([0, 0], [2, 3]), chunk([0, 2], [2, 2])],
                       [chunk([0, 0], [2, 4]), chunk([0, 0], [2, 4])]):
            with self.assertRaises(ValueError):
                _rectangles((2, 4), chunks)

    def test_chunk_escape_is_rejected(self):
        source = self.actual_checkpoint("escape")
        metadata = _metadata(source / ".metadata")
        for value in metadata.storage_data.values():
            value.relative_path = "../outside.distcp"
        (source / ".metadata").write_bytes(pickle.dumps(metadata))
        result = convert_checkpoint(source, self.root / "escape_out")
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all("escapes" in item["reason"] for item in result["unsupported"]))

    def cached_mcore_metadata(self, source, *, changed_plan=None):
        """Controlled source-schema fixture; no Megatron package is imported."""
        from torch.distributed.checkpoint.planner import WriteItem, WriteItemType, TensorWriteData
        from torch.distributed.checkpoint.metadata import MetadataIndex
        metadata = _metadata(source / ".metadata")
        name, spec = next(iter(metadata.state_dict_metadata.items()))
        module_name = "megatron.core.dist_checkpointing.strategies.torch"
        module = types.ModuleType(module_name)
        cls = type("MCoreSavePlan", (), {"__module__": module_name})
        module.MCoreSavePlan = cls
        plan = cls()
        plan.items = [WriteItem(MetadataIndex(name, spec.chunks[0].offsets), WriteItemType.SHARD,
                               tensor_data=TensorWriteData(spec.chunks[0], spec.properties, spec.size))]
        plan.storage_data = None
        plan.planner_data = {}
        plan.usable = True
        plan.mcore_data = {name: {}}
        metadata.mcore_data = {name: {}}
        if changed_plan is not None:
            changed_plan(plan, metadata)
        metadata.all_local_plans = [plan]
        modules = {module_name: module}
        parts = module_name.split(".")
        for index in range(1, len(parts)):
            parent_name = ".".join(parts[:index])
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            modules[parent_name] = parent
        for name, item in modules.items():
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(modules[parent_name], child_name, item)
        with patch.dict(sys.modules, modules):
            (source / ".metadata").write_bytes(pickle.dumps(metadata))

    def test_mcore_cached_plan_is_inert_and_regular_tensors_convert(self):
        source = self.actual_checkpoint("mcore_regular")
        self.cached_mcore_metadata(source)
        modules_before = {name for name in sys.modules if name.startswith("megatron")}
        metadata = _metadata(source / ".metadata")
        self.assertEqual(type(metadata.all_local_plans[0]).__module__, "musaprobe.distributed_checkpoint")
        result = convert_torch_dist(source, self.root / "mcore_regular_out")
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(modules_before, {name for name in sys.modules if name.startswith("megatron")})

    def test_mcore_cached_plan_unknown_fields_and_conflicts_are_rejected(self):
        changes = [lambda p, m: setattr(p, "unknown", True),
                   lambda p, m: setattr(p, "storage_data", {"unsupported": True}),
                   lambda p, m: setattr(p, "usable", "yes"),
                   lambda p, m: setattr(p, "items", ["invalid write item"]),
                   lambda p, m: setattr(p, "mcore_data", {"other": {}})]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                source = self.actual_checkpoint(f"mcore_bad_{index}")
                self.cached_mcore_metadata(source, changed_plan=change)
                with self.assertRaisesRegex(ValueError, "MCore"):
                    _metadata(source / ".metadata")

    def test_chunk_declared_shape_and_dtype_corruption_cannot_pass(self):
        for name, change in (("shape", lambda spec: setattr(spec, "size", self.torch.Size([12]))),
                             ("dtype", lambda spec: setattr(spec.properties, "dtype", self.torch.int32))):
            with self.subTest(change=name):
                source = self.actual_checkpoint(name + "_corrupt")
                metadata = _metadata(source / ".metadata")
                change(metadata.state_dict_metadata["model.weight"])
                (source / ".metadata").write_bytes(pickle.dumps(metadata))
                result = convert_torch_dist(source, self.root / (name + "_corrupt_out"))
                self.assertEqual(result["status"], "incomplete")
                self.assertTrue(any(item["name"] == "model.weight" for item in result["unsupported"]))

    def test_unknown_mcore_reformulation_is_incomplete(self):
        source = self.actual_checkpoint("mcore_unknown_layout")
        metadata = _metadata(source / ".metadata")
        metadata.mcore_data = {"model.weight": {"future_reformulation": [4, 3]}}
        (source / ".metadata").write_bytes(pickle.dumps(metadata))
        result = convert_torch_dist(source, self.root / "unknown_layout_out")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("unsupported MCore", result["unsupported"][0]["reason"])
