#!/usr/bin/env python3
"""Target-device acceptance of shared modules, checkpoint and hook lifetimes.

Run CPU first, then explicitly select MUSA. Exact comparisons are within the
same device/dtype. The intentional backward-failure run must have an error
manifest; every tensor event in both runs must still have a complete payload.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from musaprobe import PrecisionDebugger
from musaprobe.storage import atomic_json


class SharedCheckpointModel(torch.nn.Module):
    def __init__(self, reentrant=False):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                          torch.nn.Dropout(.25), torch.nn.Linear(8, 4))
        self.alias = self.shared
        self.reentrant = reentrant
        self.checkpoint_entries = 0

    def checkpointed(self, value):
        self.checkpoint_entries += 1
        return self.alias(self.shared(value))

    def forward(self, value):
        hidden = checkpoint(self.checkpointed, value, use_reentrant=self.reentrant, preserve_rng_state=True)
        return self.shared(hidden)


class DeliberateBackwardFailure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("intentional checkpoint lifecycle backward failure")


def _rng(device):
    return (torch.get_rng_state().clone(), torch.musa.get_rng_state().clone() if device == "musa" else None)


def _restore(state, device):
    torch.set_rng_state(state[0])
    if device == "musa":
        torch.musa.set_rng_state(state[1])


def _equal_rng(left, right):
    return torch.equal(left[0], right[0]) and ((left[1] is None and right[1] is None) or
           (left[1] is not None and right[1] is not None and torch.equal(left[1], right[1])))


def _module_hooks(model):
    return {name: {kind: tuple(getattr(module, kind).keys()) for kind in
                  ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks",
                   "_forward_pre_hooks_with_kwargs", "_forward_hooks_with_kwargs", "_forward_hooks_always_called")}
            for name, module in model.named_modules()}


def _read_events(probe):
    return [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]


def exercise(output, device, dtype, *, reentrant=False):
    output = Path(output)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def make_probe(name):
        return PrecisionDebugger(config={"output_dir": str(output / "captures"), "run_id": name,
            "mode": "tensor", "modules": ["shared*"], "rank": 0, "world_size": 1,
            "stack_depth": 0, "max_events": 2000, "max_tensor_bytes": 65536,
            "max_rank_bytes": 8 * 1024 * 1024})

    def run(model, initial_rng, probe=None, user_input_hook=False):
        model.zero_grad(set_to_none=True)
        model.checkpoint_entries = 0
        value = torch.arange(12, device=device, dtype=dtype).reshape(3, 4).div(16).requires_grad_()
        calls = []
        user_handle = value.register_hook(lambda grad: calls.append("user")) if user_input_hook else None
        initial_input_hooks = tuple((value._backward_hooks or {}).keys())
        _restore(initial_rng, device)
        result = model(value)
        for identity, retain in (("first", True), ("second", False)):
            scope = probe.backward_session(identity, retain_graph=retain) if probe else contextlib.nullcontext()
            with scope:
                loss = result.float().square().mean() if identity == "first" else result.float().sum()
                loss.backward(retain_graph=retain)
        if probe:
            probe.capture_parameter_grads(model)
        snapshot = {"output": result.detach().clone(), "input_gradient": value.grad.detach().clone(),
                    "parameter_gradients": {name: p.grad.detach().clone() for name, p in model.named_parameters()},
                    "rng": _rng(device), "checkpoint_entries": model.checkpoint_entries,
                    "input": value, "user_handle": user_handle, "user_calls": calls,
                    "initial_input_hooks": initial_input_hooks}
        return snapshot

    def same(name, actual, expected):
        check(name + "_output_exact", torch.equal(actual["output"], expected["output"]))
        check(name + "_input_gradient_exact", torch.equal(actual["input_gradient"], expected["input_gradient"]))
        check(name + "_all_parameter_gradients_exact", set(actual["parameter_gradients"]) == set(expected["parameter_gradients"]) and
              all(torch.equal(gradient, expected["parameter_gradients"][key]) for key, gradient in actual["parameter_gradients"].items()))
        check(name + "_cpu_device_rng_exact", _equal_rng(actual["rng"], expected["rng"]))
        check(name + "_real_checkpoint_reexecution", actual["checkpoint_entries"] >= 3 and
              actual["checkpoint_entries"] == expected["checkpoint_entries"])

    def payloads_complete(name, probe):
        records = _read_events(probe)
        check(name + "_all_events_tensor_files", bool(records) and
              all(e["state"] == "tensor" and e.get("file") and (probe.rank_dir / e["file"]).is_file() for e in records))
        check(name + "_payload_shape_and_finite", all(
              list((array := np.load(probe.rank_dir / e["file"], allow_pickle=False)).shape) == e["shape"] and np.isfinite(array).all()
              for e in records))
        return records

    torch.manual_seed(811)
    if device == "musa":
        torch.musa.manual_seed_all(811)
    baseline = SharedCheckpointModel(reentrant).to(device=device, dtype=dtype)
    observed = copy.deepcopy(baseline)
    user_calls = []
    user_module_hook = observed.shared.register_forward_hook(lambda *_: user_calls.append("module"))
    before_hooks = _module_hooks(observed)
    initial_rng = _rng(device)
    expected = run(baseline, initial_rng)
    probe = make_probe("shared_checkpoint")
    try:
        probe.start(observed)
        attached_count = len(probe._hooks)
        actual = run(observed, initial_rng, probe, user_input_hook=True)
        same("captured", actual, expected)
        check("shared_alias_is_one_module", observed.shared is observed.alias)
        check("module_hook_count_does_not_grow", len(probe._hooks) == attached_count == 2 * len(list(observed.named_modules())))
        before_close_rng = _rng(device)
        probe.close()
        check("normal_close_rng_unchanged", _equal_rng(before_close_rng, _rng(device)))
        check("normal_close_owned_hooks_empty", not probe._hooks and not probe._tensor_hooks and not probe._frames())
        check("normal_close_preserves_user_module_hook", _module_hooks(observed) == before_hooks)
        check("normal_close_preserves_user_tensor_hook", tuple((actual["input"]._backward_hooks or {}).keys()) == actual["initial_input_hooks"])
        check("user_tensor_hook_fired_twice", len(actual["user_calls"]) == 2)
        event_count = len(_read_events(probe))
        clean = run(observed, initial_rng)
        same("after_cleanup", clean, expected)
        check("closed_probe_does_not_add_events", len(_read_events(probe)) == event_count)
        check("caller_module_hook_still_runs", len(user_calls) > 0)
        records = payloads_complete("normal", probe)
        manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
        check("normal_manifest_complete", manifest["status"] == "complete" and not probe.errors)
        check("shared_calls_use_distinct_indices", {e["call_index"] for e in records if
              e["module"] == "shared" and e["kind"] == "input" and not e["recompute"]} == {0, 1, 2})
        check("recomputation_recorded_in_both_backwards", {e["backward_id"] for e in records if
              e["kind"] == "input" and e["recompute"]} == {"first", "second"})
        check("gradient_observations_have_both_backward_ids", {e["backward_id"] for e in records if
              e["kind"] in ("grad_input", "grad_output")} == {"first", "second"})
        check("all_named_parameter_gradient_files", {e.get("parameter_name") for e in records if e["kind"] == "parameter_grad"} ==
              {name for name, _ in observed.named_parameters()})
        check("saved_parameter_gradients_equal_actual_gradients", all(np.array_equal(
              np.load(probe.rank_dir / e["file"], allow_pickle=False),
              actual["parameter_gradients"][e["parameter_name"]].float().cpu().numpy())
              for e in records if e["kind"] == "parameter_grad"))
        actual["user_handle"].remove()
    finally:
        probe.close()

    failed_probe = make_probe("intentional_backward_failure")
    try:
        failed_probe.start(observed)
        observed.zero_grad(set_to_none=True)
        value = torch.ones((3, 4), device=device, dtype=dtype, requires_grad=True)
        _restore(initial_rng, device)
        result = observed(value)
        check("error_path_has_pending_tensor_hooks", bool(failed_probe._tensor_hooks))
        failure = None
        try:
            with failed_probe.backward_session("intentional-failure"):
                DeliberateBackwardFailure.apply(result).float().sum().backward()
        except RuntimeError as exc:
            failure = str(exc)
        check("original_backward_exception_propagates", failure == "intentional checkpoint lifecycle backward failure")
        check("failed_session_releases_tensor_hooks_only", not failed_probe._tensor_hooks and bool(failed_probe._hooks))
        before_close_rng = _rng(device)
        failed_probe.close()
        check("error_close_rng_unchanged", _equal_rng(before_close_rng, _rng(device)))
        check("error_close_owned_hooks_empty", not failed_probe._hooks and not failed_probe._tensor_hooks and not failed_probe._frames())
        check("error_close_preserves_user_module_hook", _module_hooks(observed) == before_hooks)
        failed_records = payloads_complete("intentional_failure", failed_probe)
        failed_manifest = json.loads((failed_probe.rank_dir / "manifest.json").read_text())
        check("failed_run_is_explicitly_not_complete", failed_manifest["status"] == "error" and
              any("backward session failed" in error for error in failed_probe.errors))
        clean_after_failure = run(observed, initial_rng)
        same("after_error_cleanup", clean_after_failure, expected)
        check("failed_closed_probe_does_not_add_events", len(_read_events(failed_probe)) == len(failed_records))
    finally:
        failed_probe.close()
        user_module_hook.remove()
    return {"status": "passed", "device": device, "dtype": str(dtype), "use_reentrant": reentrant,
            "assertions": checks, "normal_events": len(records), "intentional_failure_events": len(failed_records),
            "checkpoint_entries_per_two_backwards": expected["checkpoint_entries"],
            "expected_error_run": "intentional_backward_failure",
            "scope": "Tiny eager shared-module checkpoint model, two backward invocations. Exact same-device comparisons; intentional failure manifest remains error. No compiled graph or full-model recomputation claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--reentrant", action="store_true")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "musa":
        import torch_musa  # noqa: F401
    try:
        result = exercise(args.output, args.device, getattr(torch, args.dtype), reentrant=args.reentrant)
    except BaseException as exc:
        atomic_json(args.output / "summary.json", {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)})
        raise
    result["torch_version"] = str(torch.__version__)
    if args.device == "musa":
        result["torch_musa_version"] = str(getattr(torch_musa, "__version__", "unknown"))
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
