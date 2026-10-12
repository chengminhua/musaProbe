#!/usr/bin/env python3
"""CPU-only schedule-contract test. This does not validate Megatron/MCCL itself."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import torch
from musaprobe import PrecisionDebugger
from musaprobe.integrations.megatron import MegatronProbe, optimizer_gradient_resolver
from musaprobe.integrations import megatron_entry


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer.weight.fill_(0.5)

    def forward(self, value):
        return self.layer(value)


def exercise(directory, name, injection=None, *, distributed=False):
    model = TinyModel()
    inputs = [torch.tensor([[1., 2.]], requires_grad=True), torch.tensor([[3., 4.]], requires_grad=True)]
    observed = []
    schedules = SimpleNamespace()
    args = SimpleNamespace(curr_iteration=2, use_distributed_optimizer=distributed)
    probe = MegatronProbe("unused-test-config")
    probe.injection = injection

    def initialize(model, _args):
        probe.debugger = PrecisionDebugger(config={
            "output_dir": directory, "run_id": name, "mode": "tensor",
            "steps": [2], "microbatches": [0], "modules": ["layer"],
            "apis": ["aten.mm.default"], "rank": 0, "world_size": 1,
        }, metadata={"integration_test": True})
        probe._install_injection(model, 0)
        probe.debugger.start(model)
    probe._initialize = initialize

    def forward_step(forward_step_func, data_iterator, model, current_microbatch=None):
        observed.append(("forward", dict(probe._context)))
        return model(data_iterator), 1

    def backward_step(input_tensor, output_tensor, output_tensor_grad=None):
        observed.append(("backward", dict(probe._context)))
        output_tensor.sum().backward()
        return input_tensor.grad

    def optimizer_step():
        observed.append(("optimizer", dict(probe._context)))
        return True

    optimizer = SimpleNamespace(step=optimizer_step)
    if distributed:
        leaf = SimpleNamespace(ddp_config=SimpleNamespace(use_distributed_optimizer=True),
            model_param_gbuf_map={model.layer.weight: (0, "float32", 0)},
            _get_model_param_range_map=lambda parameter: {"param": SimpleNamespace(start=1, end=3, size=2)})
        optimizer.chained_optimizers = [SimpleNamespace(chained_optimizers=[leaf])]
        probe._parallel_coordinates = {"tensor": 0, "pipeline": 1, "data": 0}
    schedules.forward_step, schedules.backward_step = forward_step, backward_step

    def train_step(forward_step_func, data_iterator, model, optimizer):
        # Both forwards happen before either backward, as pipeline warmup can do.
        outputs = [schedules.forward_step(None, value, model[0], current_microbatch=i)[0]
                   for i, value in enumerate(inputs)]
        for value, output in zip(inputs, outputs):
            schedules.backward_step(value, output)
        for parameter in model[0].parameters():
            parameter.main_grad = parameter.grad.detach().clone()
            parameter.grad = None
        optimizer.step()
        return [value.detach().clone() for value in outputs]

    # Match musa_patch: train is defined in another module namespace and calls
    # that namespace's train_step rather than training.train_step.
    patched_globals = {"train_step": train_step}
    exec("def train(*args, **kwargs):\n    return train_step(*args, **kwargs)\n", patched_globals)
    training = SimpleNamespace(train_step=train_step, train=patched_globals["train"], get_args=lambda: args)
    probe.install(training, schedules)
    try:
        outputs = training.train(None, None, [model], optimizer)
        if distributed:
            # An excluded step/rank must not even enter parameter traversal.
            with mock.patch.object(probe.debugger, "capture_parameter_grads",
                                   side_effect=AssertionError("unselected gradient capture entered")):
                args.curr_iteration = 3
                training.train(None, None, [model], optimizer)
                args.curr_iteration = 2
                probe.debugger.config.ranks = [7]
                training.train(None, None, [model], optimizer)
                probe.debugger.config.ranks = None
    finally:
        probe.close()
    assert training.train_step is train_step and schedules.forward_step is forward_step
    assert patched_globals["train_step"] is train_step
    assert schedules.backward_step is backward_step and optimizer.step is optimizer_step
    assert [context["microbatch"] for phase, context in observed if phase == "backward"] == [0, 1] * (3 if distributed else 1)
    assert [context["phase"] for phase, context in observed] == ["forward", "forward", "backward", "backward", "optimizer"] * (3 if distributed else 1)
    rank_dir = Path(directory) / name / "rank_00000"
    events = [json.loads(line) for line in (rank_dir / "events.jsonl").read_text().splitlines()]
    assert events and all(item["step"] == 2 for item in events)
    assert {item["microbatch"] for item in events if item["phase"] == "backward"} == {0}
    gradients = [item for item in events if item["kind"] == "parameter_grad"]
    assert gradients and all(item["microbatch"] is None and item["phase"] == "optimizer" for item in gradients)
    if distributed:
        assert gradients[0]["gradient_scope"] == "optimizer_owned_shard"
        assert gradients[0]["shape"] == [2]
        assert gradients[0]["original_parameter_shape"] == [2, 2]
        assert gradients[0]["parameter_flat_range"] == [1, 3]
        assert gradients[0]["gradient_parallel_coordinates"] == {"tensor": 0, "pipeline": 1, "data": 0}
    else:
        assert gradients[0]["gradient_scope"] == "full_parameter"
    assert any(item["kind"] == "api_output" for item in events)
    assert (json.loads((rank_dir / "manifest.json").read_text())["status"]) == "complete"
    if injection is None:
        assert torch.equal(outputs[0], torch.tensor([[1.5, 1.5]]))
    elif injection["kind"] == "add":
        assert torch.allclose(outputs[0], torch.tensor([[1.6, 1.6]]))
        assert torch.equal(outputs[1], torch.tensor([[3.5, 3.5]]))
        assert probe._injection_count == 1
    else:
        assert torch.isnan(outputs[0]).all() and torch.isfinite(outputs[1]).all()
        assert any(item["stats"].get("nan_count", 0) for item in events)
    return {"run": name, "events": len(events), "parameter_gradients": len(gradients), "injected": probe._injection_count}


def exercise_shutdown(training_script):
    results = []
    for case in ("return", "exit_zero", "training_failure", "exit_nonzero", "cleanup_failure",
                 "exit_zero_cleanup_failure", "baseline", "cpu", "cuda", "uninitialized"):
        observed = []
        def synchronize():
            observed.append("synchronize")
            if case in ("cleanup_failure", "exit_zero_cleanup_failure"):
                raise RuntimeError("fake cleanup failure")
        fake_dist = SimpleNamespace(is_available=lambda: True,
            is_initialized=lambda: case != "uninitialized",
            barrier=lambda: observed.append("barrier"),
            destroy_process_group=lambda: observed.append("destroy"))
        fake_torch = SimpleNamespace(distributed=fake_dist, musa=SimpleNamespace(synchronize=synchronize),
                                     cuda=SimpleNamespace(is_initialized=lambda: case == "cuda", synchronize=synchronize))
        fake_probe = SimpleNamespace(install=lambda: fake_probe, stop=lambda: observed.append("stop"),
                                     close=lambda **kw: observed.append("close:" + kw["status"]))
        def run_training(*args, **kwargs):
            observed.append("train")
            if case in ("exit_zero", "exit_zero_cleanup_failure"):
                raise SystemExit(0)
            if case == "exit_nonzero":
                raise SystemExit(9)
            if case == "training_failure":
                raise RuntimeError("fake training failure")
        modules = {"torch": fake_torch, "torch_musa": SimpleNamespace(),
                   "musaprobe.integrations.megatron": SimpleNamespace(MegatronProbe=lambda *a, **kw: fake_probe)}
        environment = {"MUSAPROBE_ENABLED": "0" if case == "baseline" else "1",
                       "MUSAPROBE_CONFIG": "unused", "ACCELERATOR_BACKEND": "cpu"}
        raised = None
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, environment), \
             mock.patch.object(sys, "argv", ["entry", str(training_script)]), \
             mock.patch.object(sys, "path", list(sys.path)), \
             mock.patch.object(megatron_entry.runpy, "run_path", side_effect=run_training):
            if case in ("cpu", "cuda"):
                del sys.modules["torch_musa"]
            try:
                megatron_entry.main()
            except BaseException as error:
                raised = error
        if case in ("training_failure", "exit_nonzero"):
            assert observed == ["train", "stop", "close:error"], (case, observed)
        elif case in ("cleanup_failure", "exit_zero_cleanup_failure"):
            assert isinstance(raised, RuntimeError)
            assert observed == ["train", "stop", "synchronize", "close:error"], observed
        elif case == "baseline":
            assert observed == ["train", "synchronize", "barrier", "synchronize", "destroy"], observed
        elif case == "cpu":
            assert observed == ["train", "stop", "barrier", "destroy", "close:complete"], observed
        elif case == "uninitialized":
            assert observed == ["train", "stop", "close:complete"], observed
        else:
            assert observed == ["train", "stop", "synchronize", "barrier", "synchronize", "destroy", "close:complete"], observed
        if case == "exit_zero":
            assert isinstance(raised, SystemExit) and raised.code == 0
        elif case == "exit_nonzero":
            assert isinstance(raised, SystemExit) and raised.code == 9
        elif case == "training_failure":
            assert isinstance(raised, RuntimeError)
        elif case not in ("cleanup_failure", "exit_zero_cleanup_failure"):
            assert raised is None, (case, raised)
        results.append({"case": case, "order": observed})
    return results


def exercise_close_errors(directory):
    results = []
    for name, status, errors in (("collector_error", "complete", ["controlled collector failure"]),
                                 ("outer_error", "error", [])):
        probe = MegatronProbe("unused-test-config")
        probe.debugger = PrecisionDebugger(config={
            "output_dir": directory, "run_id": name, "rank": 0, "world_size": 1,
            "apis": [], "modules": ["*"],
        }).start()
        rank_dir = probe.debugger.writer.rank_dir
        probe._audit = (rank_dir / "integration.jsonl").open("w")
        probe.debugger.errors.extend(errors)
        probe.close(status=status)
        probe.close(status="complete")  # Idempotence must not erase failure.
        manifest = json.loads((rank_dir / "manifest.json").read_text())
        audit = [json.loads(line) for line in (rank_dir / "integration.jsonl").read_text().splitlines()]
        assert manifest["status"] == "error", manifest
        assert manifest["metadata"]["collector_errors"], manifest
        assert len(audit) == 1 and audit[0]["status"] == "error", audit
        if errors:
            assert manifest["metadata"]["collector_errors"] == errors
        results.append({"case": name, "manifest_status": manifest["status"],
                        "audit_status": audit[0]["status"], "errors": manifest["metadata"]["collector_errors"]})
    return results


def exercise_owned_ranges():
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    parameter.main_grad = torch.arange(6, dtype=torch.float32).view(2, 3)
    calls = []
    coordinates = {"tensor": 0, "pipeline": 1, "data": 2}
    def leaf(start=2, end=5, *, owns=True):
        def getter(value):
            calls.append(value)
            return {"param": SimpleNamespace(start=start, end=end, size=end-start)}
        return SimpleNamespace(ddp_config=SimpleNamespace(use_distributed_optimizer=True),
            model_param_gbuf_map={parameter: (0, "float32", 0)} if owns else {},
            _get_model_param_range_map=getter)
    owned = leaf()
    resolver = optimizer_gradient_resolver(
        SimpleNamespace(chained_optimizers=[SimpleNamespace(chained_optimizers=[owned])]),
        parallel_coordinates=coordinates, distributed_expected=True)
    assert calls == []  # No mapping/gradient inspection until selected capture calls it.
    result = resolver("weight", parameter)
    assert result["state"] == "tensor" and torch.equal(result["tensor"], torch.tensor([2., 3., 4.]))
    assert result["tensor"].untyped_storage().data_ptr() == parameter.main_grad.untyped_storage().data_ptr()
    assert result["metadata"]["parameter_flat_range"] == [2, 5]
    assert result["metadata"]["original_parameter_shape"] == [2, 3]
    assert result["metadata"]["gradient_parallel_coordinates"] == coordinates
    assert len(calls) == 1
    cases = {"nested_owned": result["state"]}
    for name, optimizer, expected in (
        ("not_owned", leaf(owns=False), "unsupported"),
        ("duplicate_owner", SimpleNamespace(chained_optimizers=[leaf(), leaf()]), "error"),
        ("invalid_range", leaf(-1, 7), "error"),
        ("missing_private_api", SimpleNamespace(ddp_config=SimpleNamespace(use_distributed_optimizer=True)), "unsupported"),
    ):
        result = optimizer_gradient_resolver(optimizer, parallel_coordinates=coordinates,
                                             distributed_expected=True)("weight", parameter)
        assert result["state"] == expected and result["tensor"] is None, (name, result)
        if name == "not_owned":
            assert result["metadata"]["gradient_scope"] == "not_owned_by_rank"
            assert result["metadata"]["parameter_flat_range"] is None
        cases[name] = result["state"]
    cycle = SimpleNamespace()
    cycle.chained_optimizers = [cycle]
    result = optimizer_gradient_resolver(cycle, parallel_coordinates=coordinates,
                                         distributed_expected=True)("weight", parameter)
    assert result["state"] == "error" and "cycle" in result["reason"]
    cases["cycle"] = result["state"]
    result = optimizer_gradient_resolver(SimpleNamespace(), parallel_coordinates=coordinates)("weight", parameter)
    assert result["state"] == "tensor" and result["tensor"] is parameter.main_grad
    assert result["metadata"]["gradient_scope"] == "full_parameter"
    assert result["metadata"]["parameter_flat_range"] == [0, 6]
    cases["ordinary_full"] = result["state"]
    assert torch.equal(parameter.main_grad, torch.arange(6, dtype=torch.float32).view(2, 3))
    return cases


def main():
    with tempfile.TemporaryDirectory(prefix="musaprobe-integration-") as directory:
        # Run the real file entry in a fresh interpreter, exactly as torchrun does.
        # A fake external package proves integrations/megatron.py cannot shadow it.
        root = Path(directory)
        external = root / "external"
        (external / "megatron/core").mkdir(parents=True)
        (external / "megatron/__init__.py").write_text("")
        (external / "megatron/core/__init__.py").write_text("MARKER = 'external-package-ok'\n")
        training_script = root / "fake_pretrain.py"
        training_script.write_text("from megatron.core import MARKER\nprint(MARKER)\n")
        entry = Path(__file__).resolve().parents[1] / "src/musaprobe/integrations/megatron_entry.py"
        completed = subprocess.run([sys.executable, str(entry), str(training_script)],
            env={**os.environ, "PYTHONPATH": str(external), "MUSAPROBE_ENABLED": "0"},
            text=True, capture_output=True, check=True)
        assert completed.stdout.strip() == "external-package-ok", completed.stdout
        shutdown = exercise_shutdown(training_script)
        close_errors = exercise_close_errors(directory)
        owned_ranges = exercise_owned_ranges()
        results = [exercise(directory, "normal"), exercise(directory, "owned_shard", distributed=True)]
        for kind in ("add", "nan"):
            config = {"rank": 0, "step": 2, "microbatch": 0, "module": "layer", "kind": kind}
            if kind == "add":
                config["value"] = 0.1
            results.append(exercise(directory, kind, config))
    print(json.dumps({"status": "passed", "entry_subprocess_import": "passed",
                     "scope": "CPU fake pipeline schedule; no 8-rank training", "cases": results,
                     "shutdown": shutdown, "close_errors": close_errors,
                     "owned_ranges": owned_ranges}, indent=2))


if __name__ == "__main__":
    main()
