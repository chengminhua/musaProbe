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


def events(probe):
    return [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]


def array(probe, event):
    return np.load(probe.rank_dir / event["file"], allow_pickle=False)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.number = 0

    def probe(self, **kwargs):
        self.number += 1
        result = PrecisionDebugger(config={"output_dir": self.tmp.name, "run_id": f"test{self.number}",
                                          "mode": "tensor", "stack_depth": 0, **kwargs})
        self.addCleanup(result.close)
        return result

    def test_outputs_gradients_rng_and_inplace_are_unchanged(self):
        torch.manual_seed(31)
        baseline = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(inplace=True),
                                       torch.nn.Dropout(0.25), torch.nn.Linear(8, 2))
        observed = copy.deepcopy(baseline)
        x1 = torch.randn(3, 4, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        state = torch.get_rng_state().clone()
        y1 = baseline(x1)
        y1.square().sum().backward()
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        probe = self.probe(apis=["aten.addmm.*", "aten.relu_.*"], capture_parameters=True)
        probe.start(observed)
        y2 = observed(x2)
        y2.square().sum().backward()
        probe.capture_parameter_grads(observed)
        probe.close()
        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.equal(x1.grad, x2.grad))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for first, second in zip(baseline.parameters(), observed.parameters()):
            self.assertTrue(torch.equal(first.grad, second.grad))
        kinds = {item["kind"] for item in events(probe)}
        self.assertTrue({"input", "output", "grad_input", "grad_output", "parameter", "parameter_grad", "api_input", "api_output"} <= kinds)
        self.assertFalse(probe.errors)
        self.assertTrue(all(not module._backward_hooks for module in observed.modules()))

    def test_nested_repeated_shared_module_and_kwargs(self):
        class Shared(torch.nn.Module):
            def forward(self, values, *, scale):
                return {"pair": (values["x"] * scale, [values["x"] + 1])}
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = Shared()
                self.alias = self.shared
            def forward(self, x):
                first = self.shared({"x": x}, scale=2)
                return self.alias({"x": first["pair"][0]}, scale=3)
        model = Network()
        probe = self.probe(modules=["shared"])
        probe.start(model)
        result = model(torch.ones(2, requires_grad=True))
        result["pair"][0].sum().backward()
        probe.close()
        recorded = events(probe)
        self.assertEqual({item["module"] for item in recorded}, {"shared"})
        self.assertEqual({item["call_index"] for item in recorded}, {0, 1})
        self.assertIn("output['pair'][1][0]", {item["tensor_path"] for item in recorded})
        self.assertIn("args[0]['x']", {item["tensor_path"] for item in recorded})
        self.assertTrue(all(item["gradient_semantics"] == "tensor_observation" for item in recorded if item["kind"].startswith("grad_")))

    def test_step_microbatch_filters_and_saved_backward_context(self):
        model = torch.nn.Linear(2, 2)
        probe = self.probe(steps=[2], microbatches=[0])
        probe.start(model)
        probe.set_context(step=1, microbatch=0)
        model(torch.ones(1, 2))
        probe.set_context(step=2, microbatch=1)
        model(torch.ones(1, 2))
        probe.set_context(step=2, microbatch=0)
        output = model(torch.ones(1, 2, requires_grad=True))
        probe.set_context(step=9, microbatch=None)
        output.sum().backward()
        probe.close()
        recorded = events(probe)
        self.assertEqual({item["step"] for item in recorded}, {2})
        self.assertEqual({item["microbatch"] for item in recorded}, {0})
        self.assertIn("grad_input", {item["kind"] for item in recorded})

    def test_rank_and_exclude_filters(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        probe = self.probe(exclude_modules=["0"])
        probe.start(model)
        model(torch.ones(1, 2))
        probe.close()
        self.assertNotIn("0", {item["module"] for item in events(probe)})
        skip = self.probe(rank=1, world_size=2, ranks=[0])
        skip.start(model)
        model(torch.ones(1, 2))
        skip.close()
        self.assertEqual(events(skip), [])

    def test_gradients_can_be_captured_without_forward_tensors(self):
        probe = self.probe(capture_inputs=False, capture_outputs=False)
        model = torch.nn.Linear(2, 1)
        probe.start(model)
        model(torch.ones(1, 2, requires_grad=True)).sum().backward()
        probe.close()
        self.assertEqual({item["kind"] for item in events(probe)}, {"grad_input", "grad_output"})

    def test_main_grad_precedence_and_microbatch_aggregation(self):
        probe = self.probe(microbatches=[0])
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        model.weight.main_grad = torch.full_like(model.weight, 7)
        probe.start(model)
        probe.set_context(step=3, microbatch=None)
        probe.capture_parameter_grads(model)
        probe.close()
        item, = events(probe)
        self.assertEqual(item["gradient_source"], "main_grad")
        self.assertEqual(item["gradient_scope"], "full_parameter")
        self.assertEqual(item["original_parameter_shape"], [1, 2])
        self.assertEqual(item["parameter_flat_range"], [0, 2])
        self.assertEqual(item["phase"], "optimizer")
        self.assertEqual(item["aggregation"], "accumulated")
        self.assertEqual(item["step"], 3)
        np.testing.assert_array_equal(array(probe, item), [[7, 7]])

    def test_frozen_parameters_do_not_claim_missing_gradients(self):
        probe = self.probe()
        model = torch.nn.Linear(2, 1)
        model.weight.requires_grad_(False)
        model.bias.grad = torch.ones_like(model.bias)
        probe.start(model)
        probe.capture_parameter_grads(model)
        probe.close()
        recorded = events(probe)
        self.assertEqual([item["parameter_name"] for item in recorded], ["bias"])
        self.assertEqual(recorded[0]["state"], "tensor")

    @staticmethod
    def shard_resolution(parameter, tensor, *, region=None, scope="optimizer_owned_shard", state="tensor"):
        return {"tensor": tensor, "state": state, "reason": "not owned" if tensor is None else None,
                "metadata": {"gradient_scope": scope, "original_parameter_shape": list(parameter.shape),
                             "parameter_flat_range": region,
                             "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip",
                             "gradient_source": "main_grad", "gradient_parallel_coordinates": {"dp_rank": 1, "pp_rank": 0}}}

    def test_gradient_resolver_captures_only_real_owned_shard_without_modification(self):
        model = torch.nn.Linear(3, 2, bias=False)
        gradient = torch.tensor([float("nan"), float("inf"), 2., 3., 4., float("nan")]).reshape(2, 3)
        model.weight.main_grad = gradient
        before = gradient.clone()
        probe = self.probe(apis=["*"])
        probe.start(model)
        def resolver(name, parameter):
            self.assertEqual(name, "weight")
            return self.shard_resolution(parameter, parameter.main_grad.view(-1)[2:5], region=[2, 5])
        probe.capture_parameter_grads(gradient_resolver=resolver)
        probe.close()
        item, = events(probe)
        self.assertEqual(item["kind"], "parameter_grad")
        self.assertEqual(item["state"], "tensor")
        self.assertEqual(item["shape"], [3])
        self.assertEqual(item["original_parameter_shape"], [2, 3])
        self.assertEqual(item["parameter_flat_range"], [2, 5])
        self.assertEqual(item["gradient_scope"], "optimizer_owned_shard")
        self.assertEqual(item["gradient_parallel_coordinates"], {"dp_rank": 1, "pp_rank": 0})
        self.assertEqual(item["stats"]["nan_count"], 0)
        self.assertEqual(item["stats"]["posinf_count"], 0)
        np.testing.assert_array_equal(array(probe, item), [2, 3, 4])
        torch.testing.assert_close(gradient, before, equal_nan=True, rtol=0, atol=0)

    def test_gradient_resolver_not_owned_never_falls_back_to_main_grad(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.main_grad = torch.full_like(model.weight, 99)
        probe = self.probe()
        probe.start(model)
        probe.capture_parameter_grads(gradient_resolver=lambda name, parameter: self.shard_resolution(
            parameter, None, scope="not_owned_by_rank", state="unsupported"))
        probe.close()
        item, = events(probe)
        self.assertEqual(item["state"], "unsupported")
        self.assertEqual(item["gradient_scope"], "not_owned_by_rank")
        self.assertIsNone(item["file"])
        self.assertIsNone(item["parameter_flat_range"])
        self.assertEqual(item["reason"], "not owned")

    def test_gradient_resolver_filters_before_invocation(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[1].bias.requires_grad_(False)
        probe = self.probe(steps=[2], modules=["1"], microbatches=[0])
        probe.start(model)
        calls = []
        def resolver(name, parameter):
            calls.append(name)
            return self.shard_resolution(parameter, None, scope="not_owned_by_rank", state="unsupported")
        probe.capture_parameter_grads(gradient_resolver=resolver)
        self.assertEqual(calls, [])
        probe.set_context(step=2, microbatch=None)
        probe.capture_parameter_grads(gradient_resolver=resolver)
        self.assertEqual(calls, ["1.weight"])
        probe.close()
        excluded = self.probe(rank=1, world_size=2, ranks=[0])
        excluded.start(model)
        excluded.capture_parameter_grads(gradient_resolver=resolver)
        self.assertEqual(calls, ["1.weight"])

    def test_gradient_resolver_cannot_override_event_identity_or_tensor_metadata(self):
        model = torch.nn.Linear(2, 1, bias=False)
        gradient = torch.ones_like(model.weight)
        for field, spoof in (("step", 91), ("shape", [1]), ("dtype", "spoof"), ("file", "wrong.npy"), ("stats", {})):
            with self.subTest(field=field):
                probe = self.probe()
                probe.start(model)
                def resolver(name, parameter):
                    resolution = self.shard_resolution(parameter, gradient, region=[0, 2], scope="full_parameter")
                    resolution["metadata"][field] = spoof
                    return resolution
                probe.capture_parameter_grads(gradient_resolver=resolver)
                probe.close()
                item, = events(probe)
                self.assertEqual(item["state"], "error")
                self.assertEqual(item["step"], 0)
                self.assertEqual(item["module"], "<root>")
                self.assertIsNone(item["file"])
                self.assertIn("cannot override", item["reason"])

    def test_gradient_resolver_invalid_range_or_shape_is_an_explicit_error(self):
        model = torch.nn.Linear(2, 2, bias=False)
        gradient = torch.ones(2)
        cases = [{"parameter_flat_range": [-1, 1]}, {"parameter_flat_range": [1, 5]},
                 {"parameter_flat_range": [0, 3]}, {"original_parameter_shape": [4]},
                 {"gradient_parallel_coordinates": {"dp_rank": torch.tensor(0)}},
                 {"gradient_parallel_coordinates": {}}]
        for invalid in cases:
            with self.subTest(invalid=invalid):
                probe = self.probe()
                probe.start(model)
                def resolver(name, parameter):
                    resolution = self.shard_resolution(parameter, gradient, region=[1, 3])
                    resolution["metadata"].update(invalid)
                    return resolution
                probe.capture_parameter_grads(gradient_resolver=resolver)
                probe.close()
                item, = events(probe)
                self.assertEqual(item["state"], "error")
                self.assertIsNone(item["file"])

    def test_gradient_resolver_exception_is_recorded_without_aborting_training(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.main_grad = torch.ones_like(model.weight)
        probe = self.probe()
        probe.start(model)
        def resolver(name, parameter):
            raise RuntimeError("optimizer layout unavailable")
        probe.capture_parameter_grads(gradient_resolver=resolver)
        probe.close()
        item, = events(probe)
        self.assertEqual(item["state"], "error")
        self.assertIn("optimizer layout unavailable", item["reason"])
        self.assertTrue(torch.equal(model.weight.main_grad, torch.ones_like(model.weight)))

    def test_gradient_resolver_none_tensor_is_explicitly_unsupported(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.main_grad = torch.ones_like(model.weight)
        probe = self.probe()
        probe.start(model)
        probe.capture_parameter_grads(gradient_resolver=lambda name, parameter: {
            "tensor": None, "state": "tensor", "metadata": {"gradient_scope": "not_owned_by_rank"}})
        probe.close()
        item, = events(probe)
        self.assertEqual(item["state"], "unsupported")
        self.assertEqual(item["gradient_scope"], "not_owned_by_rank")
        self.assertIsNone(item["file"])

    def test_parameter_gradients_without_model_require_an_attached_model(self):
        probe = self.probe()
        probe.start()
        with self.assertRaisesRegex(ValueError, "provide a model"):
            probe.capture_parameter_grads()

    def test_dispatch_pre_mutation_snapshot_and_call_spec(self):
        probe = self.probe(apis=["aten.add.*", "aten.add_.*"])
        probe.start()
        x = torch.tensor([1., 2.])
        with probe.phase("forward"):
            y = torch.add(x, x, alpha=2)
            x.add_(3)
        probe.close()
        recorded = events(probe)
        mutation = [item for item in recorded if item["op"] == "aten.add_.Tensor"]
        np.testing.assert_array_equal(array(probe, next(item for item in mutation if item["kind"] == "api_input")), [1, 2])
        np.testing.assert_array_equal(array(probe, next(item for item in mutation if item["kind"] == "api_output")), [4, 5])
        add = [item for item in recorded if item["op"] == "aten.add.Tensor"]
        spec = add[0]["call_spec"]
        self.assertEqual(spec["args"]["items"][0], {"tensor_path": "args[0]"})
        self.assertEqual(spec["kwargs"], {"type": "dict", "items": [["alpha", 2]]})
        self.assertTrue(all(item["call_spec"] == spec for item in add))
        self.assertEqual({item["api_call_index"] for item in add}, {0})
        self.assertEqual({item["phase"] for item in recorded}, {"forward"})
        self.assertEqual(probe._context["phase"], "unknown")
        self.assertTrue(torch.equal(y, torch.tensor([3., 6.])))

    def test_dispatch_does_not_capture_its_storage_operations(self):
        probe = self.probe(apis=["*"])
        value = torch.tensor([1.])
        probe.start()
        value + value
        probe.close()
        self.assertEqual(len(events(probe)), 3)
        self.assertEqual({item["op"] for item in events(probe)}, {"aten.add.Tensor"})

    def test_stop_removes_only_owned_hooks_and_can_restart(self):
        model = torch.nn.Linear(2, 1)
        original = model.register_forward_hook(lambda *_: None)
        self.addCleanup(original.remove)
        probe = self.probe(apis=["aten.add.*"])
        probe.start(model)
        x = torch.ones(1, 2, requires_grad=True)
        result = model(x)
        probe.stop()
        count = probe.writer.events_count
        result.sum().backward()
        torch.ones(2) + torch.ones(2)
        self.assertEqual(probe.writer.events_count, count)
        self.assertEqual(len(model._forward_hooks), 1)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        probe.stop()
        probe.step()
        probe.start(model)
        model(x)
        probe.close()
        self.assertGreater(probe.writer.events_count, count)
        with self.assertRaises(RuntimeError):
            probe.start(model)

    def test_exception_cleans_hooks_mode_and_module_stack(self):
        class Broken(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("intentional failure")
        model = Broken()
        probe = self.probe(apis=["*"])
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with probe:
                probe.start(model)
                model(torch.ones(1))
        self.assertFalse(probe.active)
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(probe._frames())
        self.assertIn("error", {item["state"] for item in events(probe)})
        manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "error")

    def test_recompute_and_unknown_api_phase_are_explicit(self):
        probe = self.probe(apis=["aten.mul.*"])
        model = torch.nn.Identity()
        probe.start(model)
        x = torch.ones(1)
        x * 2
        with probe.phase("backward"):
            model(x)
        probe.close()
        recorded = events(probe)
        self.assertTrue(all(item["phase"] == "unknown" for item in recorded if item["op"]))
        self.assertTrue(all(item["recompute"] for item in recorded if not item["op"]))

    def test_unsupported_output_and_budget_are_visible(self):
        class Odd(torch.nn.Module):
            def forward(self, x):
                return {"x": x * 2, "opaque": object()}
        probe = self.probe(max_tensor_bytes=1)
        probe.start(Odd())
        probe._models[0](torch.ones(2))
        probe.close()
        recorded = events(probe)
        self.assertEqual({item["state"] for item in recorded}, {"truncated", "unsupported"})

    def test_distributed_environment_requires_shared_run_id(self):
        config = ProbeConfig(output_dir=self.tmp.name)
        probe = PrecisionDebugger(config=config)
        self.addCleanup(probe.close)
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(ValueError, "shared run_id"):
                probe.start()
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_context_validation_and_explicit_microbatch_clear(self):
        probe = self.probe()
        probe.set_context(step=2, microbatch=5)
        probe.set_context(phase="forward")
        self.assertEqual(probe._context["microbatch"], 5)
        probe.set_context(microbatch=None)
        self.assertIsNone(probe._context["microbatch"])
        self.assertEqual(probe.step(), 3)
        for values in ({"step": -1}, {"microbatch": True}, {"phase": "guess"}, {"module": "overwrite"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                probe.set_context(**values)

    def test_checkpoint_recompute_preserves_gradients(self):
        from torch.utils.checkpoint import checkpoint
        for reentrant in (True, False):
            with self.subTest(reentrant=reentrant):
                torch.manual_seed(14)
                baseline = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Sigmoid())
                observed = copy.deepcopy(baseline)
                first = torch.randn(2, 2, requires_grad=True)
                second = first.detach().clone().requires_grad_()
                expected = checkpoint(baseline, first, use_reentrant=reentrant)
                expected.sum().backward()
                probe = self.probe()
                probe.start(observed)
                actual = checkpoint(observed, second, use_reentrant=reentrant)
                with probe.phase("backward"):
                    actual.sum().backward()
                probe.close()
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(first.grad, second.grad))
                self.assertTrue(any(item.get("recompute") for item in events(probe)))
                self.assertNotIn("error", {item["state"] for item in events(probe)})
                for a, b in zip(baseline.parameters(), observed.parameters()):
                    self.assertTrue(torch.equal(a.grad, b.grad))

    def test_manual_save_repeated_name_has_unique_call_indexes(self):
        probe = self.probe()
        probe.start()
        probe.save(torch.ones(1), "boundary")
        probe.save(torch.zeros(1), "boundary")
        probe.close()
        self.assertEqual([item["call_index"] for item in events(probe)], [0, 1])

    def test_parameters_are_snapshotted_once_at_each_selected_step(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
        probe = self.probe(steps=[2, 3], microbatches=[0], capture_parameters=True)
        probe.start(model)
        self.assertEqual(events(probe), [])
        for step in (1, 2, 3, 4):
            with torch.no_grad():
                model[0].weight.fill_(step)
            for microbatch in (0, 1):
                probe.set_context(step=step, microbatch=microbatch)
                model(torch.ones(1, 2))
        probe.close()
        snapshots = [item for item in events(probe) if item["kind"] == "parameter"]
        self.assertEqual([item["step"] for item in snapshots], [2, 3])
        self.assertEqual([item["microbatch"] for item in snapshots], [None, None])
        np.testing.assert_array_equal(array(probe, snapshots[0]), [[2, 2]])
        np.testing.assert_array_equal(array(probe, snapshots[1]), [[3, 3]])

    def test_reused_leaf_does_not_reemit_old_gradients_or_leak_hooks(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid())
        probe = self.probe()
        probe.start(model)
        reused_leaf = torch.ones(1, 2, requires_grad=True)
        for step in range(12):
            probe.set_context(step=step, microbatch=0)
            model.zero_grad(set_to_none=True)
            reused_leaf.grad = None
            model(reused_leaf).sum().backward()
            self.assertEqual(len(probe._tensor_hooks), 0)
        probe.close()
        inputs = [item for item in events(probe) if item["kind"] == "grad_input" and item["module"] == "<root>"]
        self.assertEqual([item["step"] for item in inputs], list(range(12)))

    def test_unselected_steps_do_not_accumulate_counters(self):
        model = torch.nn.Linear(2, 2)
        probe = self.probe(steps=[99], apis=["*"])
        probe.start(model)
        x = torch.ones(1, 2)
        for step in range(20):
            probe.set_context(step=step)
            model(x)
        self.assertEqual(len(probe._counts), 0)
        self.assertEqual(len(probe._api_counts), 0)
        self.assertEqual(len(probe._tensor_hooks), 0)
        probe.close()

    def test_event_budget_stops_hook_growth_and_records_truncation(self):
        model = torch.nn.Linear(2, 2)
        probe = self.probe(max_events=2)
        probe.start(model)
        x = torch.ones(1, 2, requires_grad=True)
        for step in range(5):
            probe.set_context(step=step)
            model(x).sum().backward()
        probe.close()
        recorded = events(probe)
        self.assertEqual(len(recorded), 3)
        self.assertEqual(recorded[-1]["state"], "truncated")
        self.assertEqual(probe._tensor_hooks, {})


if __name__ == "__main__":
    unittest.main()
