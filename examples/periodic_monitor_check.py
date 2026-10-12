#!/usr/bin/env python3
"""Bounded six-step CPU/MUSA acceptance for periodic module/optimizer capture.

Only steps 1 and 3 are selected. Unselected steps stay explicit gaps in offline
trends. Run in an isolated process; no device is selected implicitly.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from musaprobe import PrecisionDebugger
from musaprobe.analysis import build_trends
from musaprobe.integrations.optimizer import capture_optimizer_state
from musaprobe.storage import atomic_json


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, value):
        hidden = self.linear(value)
        return hidden, self.dropout(hidden)


def _rng(device):
    return (torch.get_rng_state().clone(), torch.musa.get_rng_state().clone()
            if device == "musa" else None)


def _restore(state, device):
    torch.set_rng_state(state[0])
    if device == "musa":
        torch.musa.set_rng_state(state[1])


def _equal_rng(left, right):
    return torch.equal(left[0], right[0]) and (
        left[1] is None and right[1] is None or
        left[1] is not None and right[1] is not None and torch.equal(left[1], right[1]))


def _numpy(value):
    return value.detach().float().cpu().numpy().copy()


def exercise(output, device="cpu", dtype=torch.float32):
    output = Path(output)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    torch.manual_seed(419)
    if device == "musa":
        torch.musa.manual_seed_all(419)
    baseline = TinyModel().to(device=device, dtype=dtype)
    observed = copy.deepcopy(baseline)
    control = torch.optim.AdamW(baseline.parameters(), lr=.01, foreach=False)
    optimizer = torch.optim.AdamW(observed.parameters(), lr=.01, foreach=False)
    probe = PrecisionDebugger(config={
        "output_dir": str(output / "captures"), "run_id": "periodic", "mode": "tensor",
        "rank": 0, "world_size": 1, "steps": list(range(6)),
        "modules": ["linear", "linear.*"], "monitor_every_n_steps": 2,
        "monitor_start_step": 1, "monitor_max_steps": 2, "stack_depth": 0,
        "max_events": 100, "max_tensor_bytes": 4096, "max_rank_bytes": 1024 * 1024,
    })
    expected = {}
    counts = []

    def step(model, opt, selected_probe=None):
        opt.zero_grad(set_to_none=True)
        # Both default RNGs are genuinely exercised, including MUSA dropout.
        cpu_random = torch.rand((), device="cpu").item()
        value = (torch.randn(2, 4, device=device, dtype=dtype) / 8 + cpu_random).requires_grad_()
        hidden, result = model(value)
        scope = selected_probe.backward_session("loss") if selected_probe else contextlib.nullcontext()
        with scope:
            result.float().square().mean().backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        if selected_probe:
            selected_probe.capture_parameter_grads(model)
        opt.step()
        states = {name + "." + key: state.detach().clone()
                  for name, parameter in model.named_parameters()
                  for key, state in opt.state[parameter].items() if isinstance(state, torch.Tensor)}
        adapter = capture_optimizer_state(selected_probe, model, opt, stage="post_step") if selected_probe else None
        return {"output": result.detach(), "hidden": hidden.detach(), "input": value.detach(),
                "input_gradient": value.grad.detach(), "gradients": gradients, "states": states,
                "parameters": {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
                "rng": _rng(device), "adapter": adapter}

    initial = _rng(device)
    probe.start(observed)
    check("install_preserves_cpu_device_rng", _equal_rng(initial, _rng(device)))
    try:
        for index in range(6):
            initial = _rng(device)
            reference = step(baseline, control)
            _restore(initial, device)
            probe.set_context(step=index, microbatch=None, phase="forward")
            actual = step(observed, optimizer, probe)
            check(f"step_{index}_output_exact", torch.equal(actual["output"], reference["output"]))
            check(f"step_{index}_input_gradient_exact", torch.equal(actual["input_gradient"], reference["input_gradient"]))
            for name in ("gradients", "parameters", "states"):
                check(f"step_{index}_{name}_exact", set(actual[name]) == set(reference[name]) and
                      all(torch.equal(value, reference[name][key]) for key, value in actual[name].items()))
            check(f"step_{index}_cpu_device_rng_exact", _equal_rng(actual["rng"], reference["rng"]))
            selected = index in (1, 3)
            check(f"step_{index}_optimizer_policy", actual["adapter"]["status"] == ("pass" if selected else "skipped") and
                  len(actual["adapter"]["captured"]) == (6 if selected else 0))
            if selected:
                expected[index] = {"input": _numpy(reference["input"]), "output": _numpy(reference["hidden"]),
                    "grad_input": _numpy(reference["input_gradient"]),
                    "parameter_grad": {k: _numpy(v) for k, v in reference["gradients"].items()},
                    "optimizer_state": {k: _numpy(v) for k, v in reference["states"].items()}}
            counts.append(probe.writer.events_count)
    finally:
        before_close = _rng(device)
        probe.close()
    check("close_preserves_cpu_device_rng", _equal_rng(before_close, _rng(device)))
    check("owned_hooks_removed", not probe._hooks and not probe._tensor_hooks)
    check("nonselected_steps_create_no_events", counts[0] == 0 and counts[1] > 0 and
          counts[1] == counts[2] and counts[3] > counts[2])
    check("max_steps_prevents_later_periodic_capture", counts[3] == counts[4] == counts[5])

    events = [json.loads(line) for line in (probe.rank_dir / "events.jsonl").read_text().splitlines()]
    manifest = json.loads((probe.rank_dir / "manifest.json").read_text())
    check("only_steps_1_and_3_saved", {event["step"] for event in events} == {1, 3})
    check("manifest_complete_without_bad_states", manifest["status"] == "complete" and
          set(manifest["states"]) == {"tensor"} and len(events) == manifest["events_count"])
    check("selected_steps_have_all_required_kinds", all(
        {e["kind"] for e in events if e["step"] == index} ==
        {"input", "output", "grad_input", "grad_output", "parameter_grad", "optimizer_state"}
        for index in (1, 3)))
    check("selected_steps_save_two_gradients_and_six_optimizer_states", all(
        len([e for e in events if e["step"] == index and e["kind"] == "parameter_grad"]) == 2 and
        len([e for e in events if e["step"] == index and e["kind"] == "optimizer_state"]) == 6
        for index in (1, 3)))
    payload_bytes = 0
    for event in events:
        path = (probe.rank_dir / (event.get("file") or "")).resolve()
        check(f"event_{event['event_id']}_file_shape_finite", bool(event.get("file")) and path.is_relative_to(probe.rank_dir) and path.is_file())
        array = np.load(path, allow_pickle=False)
        check(f"event_{event['event_id']}_payload", list(array.shape) == event["shape"] and np.isfinite(array).all())
        payload_bytes += path.stat().st_size
        reference = expected[event["step"]]
        kind = event["kind"]
        if kind in ("input", "output", "grad_input"):
            wanted = reference[kind]
        elif kind == "parameter_grad":
            wanted = reference[kind][event["parameter_name"]]
        elif kind == "optimizer_state":
            wanted = reference[kind][event["module"]]
        else:
            continue
        check(f"event_{event['event_id']}_saved_value_exact", np.array_equal(array, wanted))
    check("payload_within_one_mib", 0 < payload_bytes <= 1024 * 1024)
    check("activation_and_gradients_on_requested_device", all(str(e["device"]).startswith(device)
          for e in events if e["kind"] != "optimizer_state"))
    trends = build_trends(probe.rank_dir.parent, output=output / "trends")
    check("trend_retains_missing_step_2", bool(trends["series"]) and all(
        (point := next(p for p in series["points"] if p["step"] == 2))["state"] == "missing" and
        point["metrics"].get("mean") is None for series in trends["series"]))
    check("trend_has_two_available_steps_per_series", all(
        [p["step"] for p in series["points"] if p["state"] == "available"] == [1, 3]
        for series in trends["series"]))
    check("trend_does_not_claim_gap_free_evidence", trends["status"] == "incomplete")
    return {"status": "passed", "device": device, "dtype": str(dtype), "training_steps": 6,
            "captured_steps": [1, 3], "event_counts_after_step": counts, "events": len(events),
            "payload_bytes": payload_bytes, "trend_series": len(trends["series"]),
            "trend_status": trends["status"], "assertions": checks,
            "scope": "Tiny six-step eager model with ordinary AdamW; missing trend steps remain unknown. No long-run overhead or distributed-optimizer claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        if args.device == "musa":
            import torch_musa
            if not torch.musa.is_available():
                raise RuntimeError("Requested MUSA device is unavailable; no CPU fallback")
        result = exercise(root, args.device, getattr(torch, args.dtype))
        result["torch_version"] = str(torch.__version__)
        if args.device == "musa":
            result["torch_musa_version"] = str(getattr(torch_musa, "__version__", "unknown"))
    except Exception as exc:
        result = {"status": "error", "device": args.device, "dtype": args.dtype,
                  "reason": type(exc).__name__ + ": " + str(exc)}
        (root / "traceback.txt").write_text(traceback.format_exc())
    atomic_json(root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
