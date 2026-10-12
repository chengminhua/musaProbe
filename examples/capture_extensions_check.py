#!/usr/bin/env python3
"""Small, explicit CPU/MUSA acceptance of opt-in eager capture extensions."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from musaprobe import PrecisionDebugger
from musaprobe.compare import compare_runs
from musaprobe.storage import atomic_json


def exercise(output, device, dtype, *, async_d2h=False):
    checks = {}
    created = []
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    def records(probe):
        probe.writer.flush()
        return [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
    def stored(probe, event):
        return np.load(probe.rank_dir / event["file"], allow_pickle=False)
    def rng():
        return (torch.get_rng_state().clone(), torch.musa.get_rng_state().clone() if device == "musa" else None)
    def restore(state):
        torch.set_rng_state(state[0])
        if device == "musa":
            torch.musa.set_rng_state(state[1])
    def equal_rng(a, b):
        return torch.equal(a[0], b[0]) and (device != "musa" or torch.equal(a[1], b[1]))

    with contextlib.ExitStack() as stack:
        def probe(name, **options):
            result = PrecisionDebugger(config={"output_dir": str(output / "captures"), "run_id": name,
                                               "mode": "tensor", "stack_depth": 0, "rank": 0,
                                               "world_size": 1, "async_write": async_d2h,
                                               "async_d2h": async_d2h, **options})
            stack.callback(result.close)
            created.append(result)
            return result

        torch.manual_seed(73)
        if device == "musa":
            torch.musa.manual_seed_all(73)
        baseline = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2),
                                       torch.nn.ReLU(inplace=True), torch.nn.Linear(8, 2)).to(device=device, dtype=dtype)
        observed = copy.deepcopy(baseline)
        x0 = torch.randn(3, 4, device=device, dtype=dtype, requires_grad=True)
        x = x0.detach().clone().requires_grad_()
        before = rng()
        expected = baseline(x0)
        expected.float().square().mean().backward(retain_graph=True)
        expected.float().sum().backward()
        after = rng()
        restore(before)
        p = probe("transparent", async_write=True, async_queue_size=2, history_size=3,
                  high_level_apis=["torch.nn.functional.*"], apis=["aten.addmm.*", "aten.relu_.*"],
                  dependency_graph=True)
        p.start(observed)
        actual = observed(x)
        with p.backward_session("first", retain_graph=True):
            actual.float().square().mean().backward(retain_graph=True)
        with p.backward_session("second"):
            actual.float().sum().backward()
        p.capture_parameter_grads(observed)
        p.close()
        check("transparent_output", torch.equal(actual, expected))
        check("transparent_input_gradient", torch.equal(x.grad, x0.grad))
        check("transparent_parameter_gradients", all(torch.equal(a.grad, b.grad)
              for a, b in zip(observed.parameters(), baseline.parameters())))
        check("transparent_cpu_and_device_rng", equal_rng(rng(), after))
        check("two_backward_sessions", {e["backward_id"] for e in records(p) if e["kind"].startswith("grad_")} == {"first", "second"})
        gradient_events = [e for e in records(p) if e["kind"] in {"grad_input", "grad_output", "parameter_grad"}]
        check("gradient_snapshots_complete", bool(gradient_events) and all(e["state"] == "tensor" and e.get("file") for e in gradient_events))
        check("named_parameter_gradients_saved", len([e for e in gradient_events if e["kind"] == "parameter_grad"]) == len(list(observed.parameters())))
        if async_d2h:
            check("gradient_hooks_use_production_d2h", all(e.get("snapshot_transport") == "musa_event_d2h" for e in gradient_events))
        check("owned_hooks_removed", not p._hooks and not p._tensor_hooks and p._mode is None and p._high_mode is None)

        p = probe("slice", tensor_slices=[{"module": "sample", "kind": "output", "tensor_path": "value",
                                           "selection": [[1, 4, 2], [1, None, 1]]}])
        p.start()
        value = torch.arange(20, device=device, dtype=dtype).reshape(4, 5)
        original = value.clone()
        p.save(value, "sample")
        p.close()
        event, = records(p)
        check("slice_shape_and_values", event["shape"] == [2, 4] and np.array_equal(stored(p, event), value[1:4:2, 1:].float().cpu().numpy()))
        check("slice_original_unchanged", torch.equal(value, original))
        check("slice_normalized_scope", event["tensor_slice"] == {"original_shape": [4, 5], "selection": [
            {"start": 1, "stop": 4, "step": 2}, {"start": 1, "stop": 5, "step": 1}]})

        p = probe("replacement")
        model = torch.nn.Identity()
        value = torch.ones(2, device=device, dtype=dtype, requires_grad=True)
        p.enable_input_replacement([{"module": "<root>", "tensor_path": "args[0]", "step": 0, "rank": 0,
                                     "microbatch": 0, "value": torch.full((2,), 4., dtype=dtype)}], enabled=True)
        p.start(model)
        p.set_context(microbatch=0)
        changed = model(value)
        changed.sum().backward()
        unchanged_second = model(value)
        p.close()
        check("replacement_exact_and_one_shot", bool(torch.all(changed == 4)) and torch.equal(unchanged_second, value))
        check("replacement_preserves_source_but_changes_gradient_graph", bool(torch.all(value == 1)) and value.grad is None)
        check("replacement_intervention_recorded", any(e["kind"] == "input_replacement" and e.get("intervention") for e in records(p)))

        p = probe("async", async_write=True, async_queue_size=1)
        p.start()
        value = torch.zeros(64, device=device, dtype=dtype)
        for n in range(8):
            value.fill_(n)
            p.save(value, "mutable")
        p.close()
        check("async_immutable_snapshots", all(np.array_equal(stored(p, e), np.full(64, n)) for n, e in enumerate(records(p))))
        check("async_worker_drained", not p.writer._worker.is_alive())
        if async_d2h:
            check("production_musa_d2h_transport", all(e.get("snapshot_transport") == "musa_event_d2h"
                                                       and e.get("stats_device") == "cpu" for e in records(p)))
            check("production_slot_pool_drained", p.writer._d2h._available.qsize() == p.config.async_queue_size)
            p = probe("nondefault_d2h", async_queue_size=2)
            p.start()
            producer = torch.musa.Stream()
            with torch.musa.stream(producer):
                source = torch.empty((32, 4), device=device, dtype=dtype)
                for n in range(8):
                    source.fill_(n)
                    p.save(source.t(), "noncontiguous")
                    source.fill_(-n)
            p.close()
            producer.synchronize()
            check("production_nondefault_stream_and_noncontiguous_snapshot", all(
                np.array_equal(stored(p, e), np.full((4, 32), n)) for n, e in enumerate(records(p))))
            check("production_source_modified_after_snapshot", bool(torch.all(source == -7)))

        p = probe("nonfinite", history_size=2, stop_on_nonfinite=True, async_write=True)
        p.start()
        for value in (1., 2., 3., float("nan")):
            p.save(torch.tensor([value], device=device, dtype=dtype), "watched")
        p.writer.flush()
        trigger, = p.writer.manifest["metadata"]["nonfinite_triggers"]
        check("nonfinite_bounded_preceding_history", [e["event_id"] for e in trigger["history"]] == [1, 2])
        stopped = p.step_boundary(1)
        check("nonfinite_boundary_stop", stopped["stop"] and not p.active)
        p.close()

        p = probe("hot_update", modules=["old"])
        p.start()
        p.save(torch.ones(1, device=device), "old")
        check("hot_update_queued", p.request_config_update({"modules": ["new"]}))
        p.step_boundary(1)
        p.save(torch.ones(1, device=device), "old")
        p.save(torch.ones(1, device=device), "new")
        p.close()
        check("hot_update_boundary_versions", [(e["module"], e["config_version"]) for e in records(p)] == [("old", 0), ("new", 1)])

        def graph_run(name, alpha):
            p = probe(name, high_level_apis=["torch.add"], apis=["aten.add.*", "aten.mul.*"], dependency_graph=True)
            value = torch.ones(3, device=device, dtype=dtype)
            p.start()
            intermediate = torch.add(value, value, alpha=alpha)
            intermediate * 3
            p.close()
            events = records(p)
            producer = next(e for e in events if e["kind"] == "api_output" and e["op"] == "aten.add.Tensor")
            consumer = next(e for e in events if e["kind"] == "api_input" and e["op"] == "aten.mul.Tensor")
            check(name + "_python_parent", producer.get("api_parent", {}).get("name") == "torch.add")
            check(name + "_actual_dependency", consumer["producer_event_id"] == producer["event_id"] and consumer["tensor_id"] == producer["tensor_id"])
            return p.rank_dir.parent
        reference = graph_run("graph_reference", 1)
        repeated = graph_run("graph_repeated", 1)
        changed = graph_run("graph_changed", 2)
        normal = compare_runs(reference, repeated, output=output / "normal_comparison")
        difference = compare_runs(reference, changed, output=output / "changed_comparison")
        check("graph_normal_compare_pass", normal["overall"] == "pass")
        check("graph_intentional_difference_detected", difference["overall"] == "fail")
        for captured in created:
            captured.close()
        check("all_capture_manifests_complete", all(json.loads((captured.rank_dir / "manifest.json").read_text())["status"] == "complete" for captured in created))
        check("no_error_unsupported_truncated_or_pending_events", all(e["state"] == "tensor" for captured in created for e in records(captured)))
        return {"status": "pass", "device": device, "dtype": str(dtype), "assertions": checks,
                "assertion_count": len(checks), "normal_compare": normal["overall"],
                "intentional_difference_compare": difference["overall"],
                "async_d2h": async_d2h,
                "scope": "small single-device eager networks; " + ("MUSA event D2H and async disk" if async_d2h else "async disk only")
                         + "; replacement is intentional intervention; no overlap benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--async-d2h", action="store_true", help="Enable the production opt-in MUSA transport")
    args = parser.parse_args()
    if args.async_d2h and args.device != "musa":
        parser.error("--async-d2h requires --device musa; there is no CPU transport fallback")
    destination = args.output.expanduser().resolve() / datetime.now(timezone.utc).strftime("capture_extensions_%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if args.device == "musa":
            import torch_musa  # noqa: F401
            if not torch.musa.is_available():
                raise RuntimeError("Requested MUSA device is unavailable; no CPU fallback")
        result = exercise(destination, args.device, getattr(torch, args.dtype), async_d2h=args.async_d2h)
    except Exception as error:
        result = {"status": "error", "device": args.device, "dtype": args.dtype,
                  "reason": f"{type(error).__name__}: {error}"}
    result["output"] = str(destination)
    atomic_json(destination / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
