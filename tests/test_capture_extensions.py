import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musaprobe.capture import PrecisionDebugger
from musaprobe.config import ProbeConfig
from musaprobe.storage import RunWriter


class CaptureExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        environ = patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
        environ.start()
        self.addCleanup(environ.stop)
        self.count = 0

    def probe(self, **values):
        self.count += 1
        probe = PrecisionDebugger(config={"output_dir": self.tmp.name, "run_id": f"run{self.count}",
                                          "mode": "tensor", "stack_depth": 0, **values})
        self.addCleanup(probe.close)
        return probe

    def events(self, probe):
        probe.writer.flush()
        return [json.loads(row) for row in (probe.rank_dir / "events.jsonl").read_text().splitlines()]

    def test_slices_are_normalized_and_do_not_change_input(self):
        probe = self.probe(tensor_slices=[{"module": "sample", "kind": "output", "tensor_path": "value",
                                          "selection": [[-3, None, 2], [1, 9, None]]}])
        value = torch.arange(20.).reshape(4, 5)
        before = value.clone()
        probe.start()
        probe.save(value, "sample")
        event, = self.events(probe)
        self.assertEqual(event["tensor_slice"], {"original_shape": [4, 5], "selection": [
            {"start": 1, "stop": 4, "step": 2}, {"start": 1, "stop": 5, "step": 1}]})
        self.assertEqual(event["shape"], [2, 4])
        self.assertEqual(event["stats"]["numel"], 8)
        np.testing.assert_array_equal(np.load(probe.rank_dir / event["file"]), value[1:4:2, 1:5].numpy())
        self.assertTrue(torch.equal(value, before))

    def test_invalid_slice_is_explicit_error_and_bad_config_rejected(self):
        rule = {"module": "*", "kind": "*", "tensor_path": "*", "selection": [[0, 2, 0]]}
        with self.assertRaises(ValueError):
            self.probe(tensor_slices=[rule])
        rule["selection"] = [[None, None, 1], [None, None, 1]]
        probe = self.probe(tensor_slices=[rule])
        probe.start()
        probe.save(torch.ones(3), "bad")
        event, = self.events(probe)
        self.assertEqual(event["state"], "error")

    def test_boundary_update_is_atomic_and_versions_are_explicit(self):
        probe = self.probe(modules=["first"])
        probe.start()
        self.assertTrue(probe.request_config_update({"modules": ["second"], "apis": ["aten.add.*"]}))
        probe.save(torch.ones(1), "first")
        response = probe.step_boundary(1)
        self.assertEqual(response["applied_config_version"], 1)
        probe.save(torch.ones(1), "first")
        probe.save(torch.ones(1), "second")
        self.assertFalse(probe.request_config_update({"output_dir": "other"}))
        rejected = probe.step_boundary(2)
        self.assertFalse(rejected["accepted"])
        values = [e for e in self.events(probe) if e["kind"] == "output"]
        self.assertEqual([(e["module"], e["config_version"]) for e in values], [("first", 0), ("second", 1)])
        self.assertTrue(all(e["schema_version"] == 2 and "backward_id" in e for e in values))

    def test_boundary_calls_coordinator_on_unselected_rank_and_rejects_midforward(self):
        probe = self.probe(rank=1, world_size=2, ranks=[0])
        model = torch.nn.Identity()
        model.register_forward_pre_hook(lambda *_: probe.step_boundary(0))
        probe.start(model)
        # The user's earlier prehook is outside our module frame, so test from forward itself.
        class InForward(torch.nn.Module):
            def forward(self, value):
                probe.step_boundary(0)
                return value
        probe.stop()
        probe.start(InForward())
        with self.assertRaises(RuntimeError):
            probe._models[0](torch.ones(1))
        seen = []
        def coordinate(proposal):
            seen.append(proposal)
            return {**proposal, "accepted": True}
        probe.step_boundary(3, coordinate=coordinate)
        self.assertEqual(seen[0]["step"], 3)
        self.assertIsNone(seen[0]["update"])

    def test_periodic_monitoring_is_bounded_and_optimizer_state_has_entry(self):
        probe = self.probe(monitor_every_n_steps=2, monitor_start_step=1, monitor_max_steps=2)
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        probe.start()
        for step in range(8):
            probe.set_context(step=step)
            probe.capture_optimizer_state(opt)
        events = self.events(probe)
        self.assertEqual({event["step"] for event in events}, {1, 3})
        self.assertEqual({event["kind"] for event in events}, {"optimizer_state"})
        self.assertEqual({event["phase"] for event in events}, {"optimizer"})

    def test_queued_update_revalidates_budget_before_rank_agreement(self):
        probe = self.probe(rank=0, world_size=2, max_events=20)
        probe.start()
        self.assertTrue(probe.request_config_update({"max_events": 1}))
        probe.save(torch.ones(1), "first")
        probe.save(torch.ones(1), "second")
        proposals = []
        def coordinate(proposal):
            proposals.append(proposal)
            return {**proposal, "accepted": not bool(proposal["error"])}
        result = probe.step_boundary(1, coordinate=coordinate)
        self.assertFalse(result["accepted"])
        self.assertIsNone(proposals[0]["update"])
        self.assertIn("already recorded", proposals[0]["error"])
        self.assertEqual(probe.config.max_events, 20)
        self.assertEqual(result["applied_config_version"], 0)

    def test_separate_inflight_graphs_keep_their_own_backward_hooks(self):
        model = torch.nn.Linear(2, 1)
        probe = self.probe()
        probe.start(model)
        first = torch.ones(1, 2, requires_grad=True)
        second = torch.full((1, 2), 2., requires_grad=True)
        probe.set_context(microbatch=0)
        first_output = model(first)
        probe.set_context(microbatch=1)
        second_output = model(second)
        probe.set_context(microbatch=0)
        with probe.backward_session("first"):
            first_output.sum().backward()
        self.assertTrue(probe._tensor_hooks)
        probe.set_context(microbatch=1)
        with probe.backward_session("second"):
            second_output.sum().backward()
        self.assertFalse(probe._tensor_hooks)
        events = [e for e in self.events(probe) if e["kind"] in {"grad_input", "grad_output"}]
        self.assertEqual({(e["microbatch"], e["backward_id"]) for e in events}, {(0, "first"), (1, "second")})
        self.assertEqual(len(events), 4)
        self.assertTrue(torch.equal(first.grad, second.grad))

    def test_async_write_failure_rejects_update_before_coordination(self):
        probe = self.probe(async_write=True, rank=0, world_size=2)
        probe.start()
        self.assertTrue(probe.request_config_update({"history_size": 3}))
        with patch.object(probe.writer, "_write_payload", side_effect=OSError("controlled disk fault")):
            probe.save(torch.ones(1), "value")
            seen = []
            def coordinate(proposal):
                seen.append(proposal)
                return {**proposal, "accepted": not bool(proposal["error"])}
            response = probe.step_boundary(1, coordinate=coordinate)
        self.assertFalse(response["accepted"])
        self.assertEqual(response["applied_config_version"], 0)
        self.assertEqual(probe.config.history_size, 0)
        self.assertIn("persistence", seen[0]["error"])
        probe.close()
        self.assertFalse(probe.writer._worker.is_alive())
        self.assertEqual(json.loads((probe.rank_dir / "manifest.json").read_text())["status"], "error")

    def test_repeated_backward_sessions_keep_values_and_release_hooks(self):
        reference = torch.nn.Linear(2, 1)
        model = copy.deepcopy(reference)
        x0 = torch.ones(1, 2, requires_grad=True)
        x = x0.detach().clone().requires_grad_()
        y0 = reference(x0)
        y0.sum().backward(retain_graph=True)
        y0.sum().backward()
        probe = self.probe()
        probe.start(model)
        y = model(x)
        with probe.backward_session("first", retain_graph=True):
            y.sum().backward(retain_graph=True)
        self.assertTrue(probe._tensor_hooks)
        with probe.backward_session("second"):
            y.sum().backward()
        self.assertFalse(probe._tensor_hooks)
        self.assertTrue(torch.equal(x.grad, x0.grad))
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))
        grads = [e for e in self.events(probe) if e["kind"].startswith("grad_")]
        self.assertEqual({e["backward_id"] for e in grads}, {"first", "second"})

    def test_old_forward_preserves_config_version_after_boundary_update(self):
        model = torch.nn.Linear(1, 1)
        probe = self.probe()
        probe.start(model)
        output = model(torch.ones(1, 1, requires_grad=True))
        probe.request_config_update({"modules": ["absent"]})
        probe.step_boundary(1)
        with probe.backward_session("old"):
            output.sum().backward()
        gradients = [e for e in self.events(probe) if e["kind"].startswith("grad_")]
        self.assertTrue(gradients)
        self.assertTrue(all(e["config_version"] == 0 and e["step"] == 0 for e in gradients))

    def test_high_level_attribution_and_dependency_edges(self):
        probe = self.probe(apis=["aten.add.*", "aten.mul.*"], high_level_apis=["torch.add"], dependency_graph=True)
        a = torch.ones(2)
        probe.start()
        b = torch.add(a, a)
        c = b * 3
        probe.close()
        self.assertTrue(torch.equal(c, torch.full((2,), 6.)))
        events = self.events(probe)
        add_output = next(e for e in events if e["kind"] == "api_output" and e["op"] == "aten.add.Tensor")
        mul_input = next(e for e in events if e["kind"] == "api_input" and e["op"] == "aten.mul.Tensor")
        self.assertEqual(add_output["api_parent"]["name"], "torch.add")
        self.assertEqual(mul_input["producer_event_id"], add_output["event_id"])
        self.assertEqual(mul_input["tensor_id"], add_output["tensor_id"])
        self.assertTrue(any(e["kind"] == "python_api_input" for e in events))

    def test_mutation_creates_a_new_dependency_value(self):
        probe = self.probe(apis=["aten.add.*", "aten.add_.*", "aten.mul.*"], dependency_graph=True)
        a = torch.ones(2)
        probe.start()
        b = a + 1
        b.add_(2)
        b * 3
        probe.close()
        events = self.events(probe)
        produced = [e for e in events if e["kind"] == "api_output"]
        self.assertEqual(len({e["tensor_id"] for e in produced}), 3)
        input_event = next(e for e in events if e["op"] == "aten.mul.Tensor" and e["kind"] == "api_input")
        self.assertEqual(input_event["producer_event_id"], produced[1]["event_id"])

    def test_functional_builtin_alias_uses_public_api_name(self):
        probe = self.probe(apis=["aten.mm.*"], high_level_apis=["torch.nn.functional.linear"])
        value, weight = torch.ones(1, 2), torch.ones(3, 2)
        probe.start()
        result = torch.nn.functional.linear(value, weight)
        probe.close()
        events = self.events(probe)
        self.assertTrue(torch.equal(result, torch.full((1, 3), 2.)))
        self.assertTrue(any(e["op"] == "torch.nn.functional.linear" for e in events))
        output = next(e for e in events if e["kind"] == "api_output")
        self.assertEqual(output["api_parent"]["name"], "torch.nn.functional.linear")

    def test_replacement_is_explicit_one_shot_and_preserves_original_storage(self):
        probe = self.probe()
        model = torch.nn.Identity()
        original = torch.ones(2, requires_grad=True)
        rule = {"module": "<root>", "tensor_path": "args[0]", "step": 0, "rank": 0,
                "microbatch": 0, "value": torch.full((2,), 4.)}
        with self.assertRaises(ValueError):
            probe.enable_input_replacement([rule])
        probe.enable_input_replacement([rule], enabled=True)
        probe.start(model)
        probe.set_context(microbatch=0)
        output = model(original)
        self.assertTrue(torch.equal(output, torch.full((2,), 4.)))
        self.assertTrue(torch.equal(original, torch.ones(2)))
        self.assertIsNot(output, original)
        output.sum().backward()
        self.assertIsNone(original.grad)
        self.assertTrue(torch.equal(model(original), original))
        self.assertTrue(any(e["kind"] == "input_replacement" and e["intervention"] for e in self.events(probe)))

    def test_bad_replacement_fails_without_modifying_input_or_marking_used(self):
        probe = self.probe()
        rule = {"module": "<root>", "tensor_path": "args[0]", "step": 0, "rank": 0,
                "microbatch": 0, "value": torch.full((3,), 4.)}
        probe.enable_input_replacement([rule], enabled=True)
        probe.start(torch.nn.Identity())
        probe.set_context(microbatch=0)
        value = torch.ones(2)
        with self.assertRaises(ValueError):
            probe._models[0](value)
        self.assertFalse(probe._replacement_rules[0]["used"])
        self.assertTrue(torch.equal(value, torch.ones(2)))
        self.assertFalse(probe._frames())

    def test_nonfinite_history_and_coordinated_boundary_stop(self):
        probe = self.probe(history_size=2, stop_on_nonfinite=True, rank=1, world_size=2)
        probe.start()
        for value in (1., 2., 3., float("nan")):
            probe.save(torch.tensor([value]), "watched")
        self.assertTrue(probe.stop_requested)
        self.assertEqual(len(probe.recent_history), 2)
        trigger, = probe.writer.manifest["metadata"]["nonfinite_triggers"]
        self.assertEqual([e["event_id"] for e in trigger["history"]], [1, 2])
        declined = probe.step_boundary(1)
        self.assertFalse(declined["stop"])
        self.assertTrue(probe.active)
        stopped = probe.step_boundary(1, coordinate=lambda value: {**value, "accepted": True})
        self.assertTrue(stopped["stop"])
        self.assertFalse(probe.active)

    def test_async_snapshot_is_immutable_and_close_drains(self):
        probe = self.probe(async_write=True, async_queue_size=1)
        probe.start()
        x = torch.zeros(64)
        for n in range(8):
            x.fill_(n)
            probe.save(x, "state")
        probe.close()
        events = self.events(probe)
        self.assertEqual(len(events), 8)
        for n, event in enumerate(events):
            np.testing.assert_array_equal(np.load(probe.rank_dir / event["file"]), np.full((64,), n))
        self.assertFalse(probe.writer._worker.is_alive())

    def test_async_write_failure_is_manifest_error_and_does_not_hang(self):
        probe = self.probe(async_write=True, async_queue_size=1)
        probe.start()
        with patch.object(probe.writer, "_write_payload", side_effect=OSError("disk fault")):
            probe.save(torch.ones(1), "state")
            probe.writer.flush()
        probe.close()
        event, = self.events(probe)
        self.assertEqual(event["state"], "error")
        manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "error")
        self.assertIn("storage_errors", manifest["metadata"])

    def test_combined_observation_keeps_output_gradients_and_rng(self):
        torch.manual_seed(7)
        first = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.3), torch.nn.ReLU(inplace=True))
        second = copy.deepcopy(first)
        x0 = torch.ones(2, 2, requires_grad=True)
        x = x0.detach().clone().requires_grad_()
        rng = torch.get_rng_state().clone()
        expected = first(x0)
        expected.sum().backward()
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        probe = self.probe(async_write=True, history_size=3, high_level_apis=["torch.nn.functional.*"],
                           apis=["aten.addmm.*", "aten.relu_.*"], dependency_graph=True)
        probe.start(second)
        output = second(x)
        with probe.backward_session():
            output.sum().backward()
        probe.close()
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(torch.equal(x.grad, x0.grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_backward_session_exception_cleans_even_retained_hooks(self):
        probe = self.probe()
        probe.start(torch.nn.Linear(1, 1))
        probe._models[0](torch.ones(1, 1, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "backward failure"):
            with probe.backward_session("failed", retain_graph=True):
                raise RuntimeError("backward failure")
        self.assertFalse(probe._tensor_hooks)
        self.assertIsNone(probe._backward_session)
        with self.assertRaises(ValueError):
            with probe.backward_session("failed"):
                pass

    def test_backward_identity_cannot_be_reused_after_microbatch_roundtrip(self):
        probe = self.probe()
        probe.start()
        for microbatch in (0, 1):
            probe.set_context(microbatch=microbatch)
            with probe.backward_session("loss"):
                pass
        probe.set_context(microbatch=0)
        with self.assertRaisesRegex(ValueError, "already used"):
            with probe.backward_session("loss"):
                pass
        probe.set_context(step=1)
        with probe.backward_session("loss"):
            pass

    def test_unobserved_mutation_and_inference_graph_are_explicit_unknown(self):
        probe = self.probe(apis=["aten.add.*", "aten.mul.*"], dependency_graph=True)
        value = torch.ones(2)
        probe.start()
        result = value + 1
        result.add_(3)  # excluded operation invalidates the last observed version
        result * 2
        with torch.inference_mode():
            torch.ones(2) * 2
        probe.close()
        inputs = [e for e in self.events(probe) if e["op"] == "aten.mul.Tensor" and e["kind"] == "api_input"]
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all(e["producer_event_id"] is None and e["graph_unknown_reason"] for e in inputs))

    def test_native_fp8_adapter_preserves_bits_and_disabled_is_unsupported(self):
        from musaprobe.native_fp8 import fp8_snapshot
        value = torch.tensor([1., -2., .5]).to(torch.float8_e4m3fn)
        disabled = self.probe()
        disabled.start()
        disabled.save(value, "fp8")
        self.assertEqual(self.events(disabled)[0]["state"], "unsupported")
        probe = PrecisionDebugger(config={"output_dir": self.tmp.name, "run_id": "fp8",
                                          "mode": "tensor", "capture_fp8": True, "async_write": True},
                                  fp8_adapter=fp8_snapshot)
        self.addCleanup(probe.close)
        probe.start()
        before = value.view(torch.uint8).clone()
        probe.save(value, "fp8")
        probe.close()
        event, = self.events(probe)
        self.assertEqual(event["fp8"]["scale_mode"], "unscaled_native_dtype")
        self.assertEqual(event["stored_dtype"], "uint8")
        self.assertEqual(event["stats"]["finite_count"], 3)
        np.testing.assert_array_equal(np.load(probe.rank_dir / event["file"]), before.numpy())
        self.assertTrue(torch.equal(value.view(torch.uint8), before))

    def test_async_event_flush_failure_still_closes_worker_and_marks_error(self):
        from unittest.mock import Mock
        probe = self.probe(async_write=True)
        probe.start()
        real_stream = probe.writer._stream
        wrapped = Mock(wraps=real_stream)
        wrapped.flush.side_effect = OSError("event flush failure")
        probe.writer._stream = wrapped
        probe.save(torch.ones(1), "state")
        probe.close()
        self.assertFalse(probe.writer._worker.is_alive())
        self.assertEqual(json.loads((probe.rank_dir / "manifest.json").read_text())["status"], "error")


if __name__ == "__main__":
    unittest.main()
