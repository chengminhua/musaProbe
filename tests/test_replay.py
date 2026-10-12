from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from musaprobe.replay import PURE_OPS, replay_call


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.rank = Path(self.temp.name) / "rank_00000"
        self.rank.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, left=None, right=None, output=None, *, op="aten.add.Tensor", dtype=None, alpha=None):
        left = np.array([1., 2.], np.float32) if left is None else np.asarray(left)
        right = np.array([3., 4.], np.float32) if right is None else np.asarray(right)
        output = left + right if output is None else np.asarray(output)
        spec = {"args": {"type": "tuple", "items": [{"tensor_path": "$args[0]"}, {"tensor_path": "$args[1]"}]},
                "kwargs": {"type": "dict", "items": [["alpha", alpha]] if alpha is not None else []}}
        events = []
        for i, (value, kind, tensor_path) in enumerate(((left, "api_input", "$args[0]"),
                                                       (right, "api_input", "$args[1]"),
                                                       (output, "api_output", "$output"))):
            np.save(self.rank / f"tensor_{i}.npy", value, allow_pickle=False)
            events.append({"schema_version": 1, "event_id": i, "step": 0, "microbatch": 0,
                           "phase": "forward", "module": "test", "call_index": 0, "api_call_index": 0,
                           "op": op, "kind": kind, "tensor_path": tensor_path, "call_spec": spec,
                           "shape": list(value.shape), "dtype": dtype or "torch." + value.dtype.name,
                           "file": f"tensor_{i}.npy", "state": "tensor", "stats": {"numel": value.size}})
        self.write_events(events)
        return events

    def write_events(self, events):
        (self.rank / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def test_pure_add_replay_pass(self):
        self.fixture()
        result = replay_call(self.rank, 2)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["max_abs_error"], 0)
        self.assertEqual(result["shape"], [2])

    def test_parameter_gradient_cannot_select_an_unrelated_api_replay(self):
        events = self.fixture()
        events.append({**events[2], "event_id": 3, "kind": "parameter_grad",
                       "gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [4],
                       "parameter_flat_range": [0, 2], "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip",
                       "gradient_parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 0}})
        self.write_events(events)
        result = replay_call(self.rank, 3)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("api_input/api_output", result["reason"])

    def test_api_replay_preserves_shard_metadata_and_claims_only_local_call(self):
        events = self.fixture()
        layout = {"gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [4],
                  "parameter_flat_range": [0, 2], "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip",
                  "gradient_parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 0}}
        events[2].update(layout)
        self.write_events(events)
        result = replay_call(self.rank, 2)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["replay_scope"], "captured_api_call")
        self.assertEqual(result["gradient_layout"], layout)
        self.assertIn("does not restore", result["reason"])

    def test_scalar_kwargs_and_selection_by_input(self):
        self.fixture(output=np.array([7., 10.], np.float32), alpha=2)
        self.assertEqual(replay_call(self.rank, 0)["status"], "pass")

    def test_controlled_mismatch_and_nonfinite(self):
        self.fixture(output=np.array([4., 7.], np.float32))
        self.assertEqual(replay_call(self.rank, 2)["status"], "mismatch")
        self.fixture(left=np.array([np.nan]), right=np.array([1.]))
        result = replay_call(self.rank, 2)
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["finite"])

    def test_bfloat16_snapshot_and_float32_reference(self):
        self.fixture(dtype="torch.bfloat16")
        self.assertEqual(replay_call(self.rank, 2)["status"], "pass")
        self.assertEqual(replay_call(self.rank, 2, reference_dtype="float32")["status"], "pass")

    def test_exact_int64_detects_unit_difference(self):
        self.fixture(left=np.array([2**62], np.int64), right=np.array([1], np.int64), output=np.array([2**62], np.int64))
        result = replay_call(self.rank, 2)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["max_abs_error"], 1)

    def test_empty_tensors_are_replayable(self):
        self.fixture(left=np.array([], np.float32), right=np.array([], np.float32))
        self.assertEqual(replay_call(self.rank, 2)["status"], "pass")

    def test_whitelist_excludes_mutable_random_collective_and_custom(self):
        forbidden = ["aten.add_.Tensor", "aten.rand.default", "c10d.allreduce_.default", "custom.magic.default", "__import__.os.system"]
        for op in forbidden:
            self.assertNotIn(op, PURE_OPS)
            self.fixture(op=op)
            self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")

    def test_missing_input_and_multiple_outputs_not_pass(self):
        events = self.fixture()
        self.write_events(events[1:])
        self.assertNotEqual(replay_call(self.rank, 2)["status"], "pass")
        events = self.fixture()
        events.append({**events[2], "event_id": 3, "tensor_path": "$output.extra"})
        self.write_events(events)
        self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")

    def test_statistics_and_truncated_not_replayable(self):
        for state in ("statistics", "truncated", "unsupported", "error"):
            events = self.fixture()
            events[0]["state"] = state
            self.write_events(events)
            self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")

    def test_missing_spec_unknown_objects_and_duplicate_paths(self):
        events = self.fixture()
        for event in events:
            event.pop("call_spec")
        self.write_events(events)
        self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")
        events = self.fixture()
        events[2]["call_spec"] = {"args": {"type": "object", "code": "never executed"}, "kwargs": {"type": "dict", "items": []}}
        self.write_events(events)
        self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")
        events = self.fixture()
        events[1]["tensor_path"] = events[0]["tensor_path"]
        self.write_events(events)
        self.assertEqual(replay_call(self.rank, 2)["status"], "unsupported")

    def test_other_invocation_cannot_supply_missing_input(self):
        events = self.fixture()
        events[1]["api_call_index"] = 1
        self.write_events(events)
        self.assertNotEqual(replay_call(self.rank, 2)["status"], "pass")

    def test_input_payload_dtype_must_match_metadata(self):
        self.fixture()
        np.save(self.rank / "tensor_0.npy", np.array([1, 2], np.int64))
        self.assertNotEqual(replay_call(self.rank, 2)["status"], "pass")

    def test_actual_output_dtype_must_match_recording_in_same_mode(self):
        events = self.fixture(output=np.array([4, 6], np.int64))
        self.assertEqual(events[2]["dtype"], "torch.int64")
        self.assertNotEqual(replay_call(self.rank, 2)["status"], "pass")

    def test_float32_reference_cannot_convert_recorded_integer_output(self):
        self.fixture(output=np.array([4, 6], np.int64))
        self.assertNotEqual(replay_call(self.rank, 2, reference_dtype="float32")["status"], "pass")

    def test_unsafe_file_reference_rejected(self):
        for value in ("../outside.npy", "/etc/passwd"):
            events = self.fixture()
            events[0]["file"] = value
            self.write_events(events)
            self.assertNotEqual(replay_call(self.rank, 2)["status"], "pass")

    def test_duplicate_or_missing_event_id_rejected(self):
        events = self.fixture()
        with self.assertRaises(ValueError):
            replay_call(self.rank, 999)
        events[0]["event_id"] = 2
        self.write_events(events)
        with self.assertRaises(ValueError):
            replay_call(self.rank, 2)


if __name__ == "__main__":
    unittest.main()
