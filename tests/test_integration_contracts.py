"""CPU schedule and ownership contracts; these are not Megatron device tests."""
import copy
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musaprobe import PrecisionDebugger
from musaprobe.distributed import SafeBoundaryCoordinator, BoundaryCoordinationError
from musaprobe.environment import capture_training_config
from musaprobe.integrations.communication import observed_all_reduce
from musaprobe.integrations.megatron import MegatronProbe, optimizer_gradient_resolver


class IntegrationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        environment = patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1",
            "MUSAPROBE_CONTROL_GROUP": "0", "MUSAPROBE_HOT_CONFIG": "",
            "MUSAPROBE_VALIDATE_PP_RANKS": "0", "MUSAPROBE_SOURCE_MANIFEST": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def records(self, debugger):
        debugger.writer.flush()
        return [json.loads(line) for line in (debugger.rank_dir / "events.jsonl").read_text().splitlines()]

    def test_gradient_resolver_is_independent_of_training_environment(self):
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.main_grad = torch.arange(4.).view(2, 2)
        resolver = optimizer_gradient_resolver(types.SimpleNamespace(), parallel_coordinates={"data": 0})
        with patch("musaprobe.environment.capture_training_config", side_effect=AssertionError("per-parameter environment capture")):
            result = resolver("weight", model.weight)
        self.assertEqual(result["state"], "tensor")
        self.assertIs(result["tensor"], model.weight.main_grad)
        self.assertNotIn("training", result["metadata"])

    def test_initialize_captures_environment_once_at_the_declared_boundary(self):
        path = self.directory / "config.json"
        path.write_text(json.dumps({"output_dir": str(self.directory), "run_id": "initialize", "rank": 0, "world_size": 1}))
        probe = MegatronProbe(path)
        self.addCleanup(probe.close)
        core = types.ModuleType("megatron.core")
        core.parallel_state = types.SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0, get_data_parallel_rank=lambda: 0)
        args = types.SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                     data_parallel_size=1, bf16=True, fp16=False, num_layers=2)
        with patch.dict(sys.modules, {"megatron": types.ModuleType("megatron"), "megatron.core": core}), \
             patch("torch.distributed.get_rank", return_value=0), \
             patch("musaprobe.environment.capture_training_config", wraps=capture_training_config) as environment:
            probe._initialize(torch.nn.Linear(2, 2), args)
        environment.assert_called_once()
        metadata = probe.debugger.writer.manifest["metadata"]
        self.assertEqual(metadata["training"]["num_layers"], 2)
        self.assertEqual(metadata["precision"]["bf16"], True)
        self.assertEqual(metadata["environment_point"], "probe_initialization_before_first_train_step")
        self.assertIsNone(metadata["input_state"])
        self.assertTrue(metadata["random_state"]["torch_cpu"])

    def test_vpp_inflight_graphs_keep_chunk_microbatch_and_backward_identity(self):
        chunks = [torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)) for _ in range(2)]
        reference = copy.deepcopy(chunks)
        inputs = [torch.tensor([[1., 2.]], requires_grad=True), torch.tensor([[3., 4.]], requires_grad=True),
                  torch.tensor([[5., 6.]], requires_grad=True), torch.tensor([[7., 8.]], requires_grad=True)]
        expected_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        expected = [reference[chunk](value) for value, (chunk, _) in zip(expected_inputs, order)]
        for index in (1, 0, 3, 2):
            expected[index].sum().backward()
        probe = MegatronProbe("unused")
        self.addCleanup(probe.close)
        def initialize(model, args):
            probe.debugger = PrecisionDebugger(config={"output_dir": str(self.directory), "run_id": "vpp",
                "rank": 0, "world_size": 1, "mode": "tensor", "stack_depth": 0})
            probe.debugger.start(model)
        probe._initialize = initialize
        args = types.SimpleNamespace(curr_iteration=4, use_distributed_optimizer=False)
        schedules = types.SimpleNamespace()
        seen = []
        def forward_step(data_iterator, model, current_microbatch=None):
            return model(data_iterator), 1
        def backward_step(input_tensor, output_tensor, output_tensor_grad=None):
            seen.append(dict(probe._context))
            output_tensor.sum().backward()
            return input_tensor.grad
        schedules.forward_step, schedules.backward_step = forward_step, backward_step
        original_step = lambda: True
        optimizer = types.SimpleNamespace(step=original_step)
        def train_step(model, optimizer):
            actual = [schedules.forward_step(value, model[chunk], current_microbatch=mb)[0]
                      for value, (chunk, mb) in zip(inputs, order)]
            for index in (1, 0, 3, 2):
                schedules.backward_step(inputs[index], actual[index])
            optimizer.step()
            return actual
        training = types.SimpleNamespace(train_step=train_step, get_args=lambda: args)
        probe.install(training, schedules)
        actual = training.train_step(chunks, optimizer)
        self.assertEqual([(item["virtual_pipeline"], item["microbatch"]) for item in seen],
                         [(1, 0), (0, 0), (1, 1), (0, 1)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(all(torch.equal(a.grad, b.grad) for a, b in zip(inputs, expected_inputs)))
        self.assertTrue(all(torch.equal(a[0].weight.grad, b[0].weight.grad) for a, b in zip(chunks, reference)))
        events = self.records(probe.debugger)
        gradients = [e for e in events if e["kind"] == "grad_output" and e["module"] in {"chunks.0.0", "chunks.1.0"}]
        self.assertEqual(len(gradients), 4)
        self.assertEqual({(e["virtual_pipeline"], e["microbatch"]) for e in gradients}, set(order))
        self.assertEqual(len({e["backward_id"] for e in gradients}), 4)
        self.assertFalse(probe.debugger._tensor_hooks)
        self.assertIs(optimizer.step, original_step)
        probe.close()
        self.assertIs(training.train_step, train_step)
        self.assertIs(schedules.backward_step, backward_step)

    def test_collective_wrapper_restores_context_on_failure(self):
        debugger = PrecisionDebugger(config={"output_dir": str(self.directory), "run_id": "communication", "mode": "tensor"})
        self.addCleanup(debugger.close)
        debugger.start()
        debugger.set_context(step=2, microbatch=1, phase="forward", virtual_pipeline=3)
        before = dict(debugger._context)
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce", side_effect=RuntimeError("controlled collective failure")):
            with self.assertRaisesRegex(RuntimeError, "controlled collective failure"):
                observed_all_reduce(debugger, torch.ones(2), group_id="world", sequence=0)
        self.assertEqual(debugger._context, before)
        self.assertEqual(self.records(debugger)[0]["communication"]["stage"], "input")
        self.assertEqual(self.records(debugger)[-1]["state"], "error")
        debugger.close()
        self.assertEqual(json.loads((debugger.rank_dir / "manifest.json").read_text())["status"], "error")

    def test_vpp_reordered_chunks_and_partial_backward_matches_are_rejected(self):
        models = [torch.nn.Identity(), torch.nn.Identity()]
        probe = MegatronProbe("unused")
        self.addCleanup(probe.close)
        probe._model_chunks = {id(model): index for index, model in enumerate(models)}
        probe._capture_model = torch.nn.ModuleList(models)
        probe.debugger = PrecisionDebugger(config={"output_dir": str(self.directory), "run_id": "ambiguous",
            "rank": 0, "world_size": 1, "capture_gradients": False, "stack_depth": 0}).start(probe._capture_model)
        def forward_step(data_iterator, model, current_microbatch=None):
            return model(data_iterator), 1
        def backward_step(output_tensor):
            return "reached original"
        schedules = types.SimpleNamespace(forward_step=forward_step, backward_step=backward_step)
        args = types.SimpleNamespace(curr_iteration=0, use_distributed_optimizer=False)
        def train_step(model, optimizer):
            optimizer.step()
        training = types.SimpleNamespace(train_step=train_step, get_args=lambda: args)
        optimizer = types.SimpleNamespace(step=lambda: None)
        probe.install(training, schedules)
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            training.train_step(list(reversed(models)), optimizer)
        known = schedules.forward_step(torch.ones(2), models[0], current_microbatch=0)[0]
        with self.assertRaisesRegex(RuntimeError, "Cannot associate"):
            schedules.backward_step([known, torch.zeros(2)])
        shared = torch.ones(2)
        schedules.forward_step(shared, models[0], current_microbatch=0)
        with self.assertRaisesRegex(RuntimeError, "shared by different"):
            schedules.forward_step(shared, models[1], current_microbatch=1)
        self.assertEqual(probe._context["phase"], "unknown")
        probe.close()
        self.assertEqual(json.loads((probe.debugger.rank_dir / "manifest.json").read_text())["status"], "error")

    def test_duplicate_backward_leaves_do_not_create_false_unknowns(self):
        model = torch.nn.Identity()
        probe = MegatronProbe("unused")
        self.addCleanup(probe.close)
        probe._model_chunks = {id(model): 0}
        probe.debugger = PrecisionDebugger(config={"output_dir": str(self.directory), "run_id": "duplicate",
            "rank": 0, "world_size": 1, "capture_gradients": False}).start(model)
        def forward_step(data_iterator, model, current_microbatch=None):
            return model(data_iterator), 1
        def backward_step(output_tensor):
            return dict(probe._context)
        schedules = types.SimpleNamespace(forward_step=forward_step, backward_step=backward_step)
        def train_step(model, optimizer):
            pass
        training = types.SimpleNamespace(train_step=train_step)
        probe.install(training, schedules)
        value = schedules.forward_step(torch.ones(2), model, current_microbatch=3)[0]
        context = schedules.backward_step([value, value])
        self.assertEqual(context["microbatch"], 3)
        self.assertEqual(context["phase"], "backward")

    def test_coordinator_failure_poison_and_external_group_ownership(self):
        group = object()
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_backend", return_value="gloo"), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=RuntimeError("controlled timeout")) as gather, \
             patch("torch.distributed.destroy_process_group") as destroy:
            coordinator = SafeBoundaryCoordinator(group=group)
            proposal = {"step": 1, "config_version": 0, "stop": False, "update": None}
            with self.assertRaises(BoundaryCoordinationError):
                coordinator(proposal)
            with self.assertRaisesRegex(BoundaryCoordinationError, "previously failed"):
                coordinator(proposal)
            self.assertEqual(gather.call_count, 1)
            coordinator.close()
            destroy.assert_not_called()

    def test_environment_does_not_initialize_device_or_advance_rng(self):
        fake = types.SimpleNamespace(is_initialized=lambda: False,
            get_rng_state=lambda: self.fail("uninitialized device queried"))
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with patch.object(torch, "musa", fake, create=True):
            value = capture_training_config(inputs={"bf16_view": torch.arange(12.).to(torch.bfloat16).view(3, 4).t()})
        self.assertIsNone(value["random_state"]["musa"])
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(value["input_state"]["bf16_view"]["shape"], [4, 3])
        self.assertEqual(value["environment_scope"]["inputs"], "explicit_named_tensors_only")


if __name__ == "__main__":
    unittest.main()
