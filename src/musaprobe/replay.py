"""Replay an explicitly selected pure ATen call from local tensor snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PURE_OPS = (
    "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
    "aten.mm.default", "aten.bmm.default", "aten.matmul.default", "aten.relu.default",
    "aten.sigmoid.default", "aten.tanh.default", "aten.sin.default", "aten.cos.default",
    "aten.pow.Tensor_Scalar",
    "aten.linear.default", "aten.gelu.default", "aten.layer_norm.default", "aten.softmax.int",
)
_GROUP = ("step", "microbatch", "phase", "module", "call_index", "op", "api_call_index", "recompute")


def load_captured_call(rank_dir, event_id):
    """Load a whitelisted call as CPU tensors, without executing serialized code.

    Returns ``(op_name, args, kwargs)``. This does not restore training state or
    captured backward seeds. Incomplete and ambiguous captures are rejected.
    """
    import torch
    from .compare import _load_tensor
    root = Path(rank_dir).expanduser().resolve()
    if (root / "events.jsonl").stat().st_size > 128 * 1024**2:
        raise ValueError("Captured call metadata exceeds 128 MiB read budget")
    with (root / "events.jsonl").open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    matches = [e for e in events if e.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError("event_id must identify exactly one captured event")
    selected = matches[0]
    op = selected.get("op")
    if selected.get("kind") not in {"api_input", "api_output"} or op not in PURE_OPS:
        raise ValueError("Selection is not a whitelisted captured API call")
    identity = lambda e: tuple(e.get(k, False if k == "recompute" else None) for k in _GROUP)
    invocation = [e for e in events if identity(e) == identity(selected)]
    inputs = [e for e in invocation if e.get("kind") == "api_input"]
    outputs = [e for e in invocation if e.get("kind") == "api_output"]
    if not inputs or len(outputs) != 1 or any(e.get("state") != "tensor" for e in inputs + outputs):
        raise ValueError("A complete single-output tensor capture is required")
    spec = selected.get("call_spec") or outputs[0].get("call_spec")
    if not spec:
        raise ValueError("Capture has no scalar/container call specification")
    values = {}
    for event in inputs:
        key = event["tensor_path"]
        if key in values:
            raise ValueError("Duplicate tensor path")
        relative = event.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Invalid tensor file path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 256 * 1024**2:
            raise ValueError("Tensor escapes capture or exceeds the read budget")
        name = event["dtype"].removeprefix("torch.")
        if name not in {"float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"}:
            raise ValueError("Unsupported logical dtype")
        values[key] = torch.from_numpy(_load_tensor(root, event).copy()).to(getattr(torch, name))
    def decode(value, depth=0):
        if depth > 40:
            raise ValueError("Call specification nesting exceeds limit")
        if value is None or type(value) in (bool, int, float, str):
            return value
        if not isinstance(value, dict):
            raise ValueError("Invalid call specification")
        if set(value) == {"tensor_path"}:
            return values[value["tensor_path"]]
        items = value.get("items")
        if value.get("type") == "tuple":
            return tuple(decode(x, depth+1) for x in items)
        if value.get("type") == "list":
            return [decode(x, depth+1) for x in items]
        if value.get("type") == "dict":
            return {k: decode(v, depth+1) for k, v in items}
        raise ValueError("Unsupported call specification object")
    args, kwargs = decode(spec["args"]), decode(spec["kwargs"])
    if not isinstance(args, (tuple, list)) or not isinstance(kwargs, dict):
        raise ValueError("Call arguments must be a sequence and keyword dictionary")
    return op, args, kwargs


def replay_call(rank_dir, event_id, *, device="cpu", reference_dtype="same", atol=None, rtol=None):
    """Return pass/mismatch/unsupported/error; never execute a serialized function."""
    root = Path(rank_dir).expanduser().resolve()
    result = {"schema_version": 1, "rank_dir": str(root), "event_id": event_id,
              "device": device, "reference_dtype": reference_dtype, "status": "unsupported"}
    if device not in ("cpu", "musa", "cuda") or reference_dtype not in ("same", "float32"):
        raise ValueError("device must be cpu/musa/cuda; reference_dtype must be same/float32")
    for name, tolerance in (("atol", atol), ("rtol", rtol)):
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    events = []
    with (root / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                events.append(json.loads(line))
    matches = [event for event in events if event.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError("event_id must identify exactly one captured event")
    selected = matches[0]
    result["replay_scope"] = "captured_api_call"
    if selected.get("kind") not in {"api_input", "api_output"}:
        result["reason"] = "Select an api_input/api_output event; a parameter-gradient shard is not a replayable full parameter or API output."
        return result
    from .compare import _GRADIENT_LAYOUT_FIELDS
    layout = {key: selected[key] for key in _GRADIENT_LAYOUT_FIELDS if key in selected}
    if layout:
        result["gradient_layout"] = layout
    op_name = selected.get("op")
    result["op"] = op_name
    if op_name not in PURE_OPS:
        result["reason"] = "Only the documented pure ATen whitelist is replayable; mutable/random/collective/custom calls are excluded."
        return result
    def identity(event):
        return tuple(event.get(key, False if key == "recompute" else None) for key in _GROUP)
    invocation = [event for event in events if identity(event) == identity(selected)]
    inputs = [e for e in invocation if e.get("kind") == "api_input"]
    outputs = [e for e in invocation if e.get("kind") == "api_output"]
    if len(outputs) != 1 or not inputs:
        result["reason"] = "Replay requires all input tensors and one tensor output from the same invocation."
        return result
    if any(e.get("state") != "tensor" for e in inputs + outputs):
        result["reason"] = "Statistics-only, truncated or unsupported inputs/outputs cannot be replayed."
        return result
    spec = selected.get("call_spec") or outputs[0].get("call_spec")
    if not spec:
        result["reason"] = "Capture does not contain the scalar/container call specification."
        return result
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        result.update(status="unsupported", reason="No CUDA device is available")
        return result
    if device == "musa":
        try:
            import torch_musa  # noqa: F401
        except ImportError:
            result.update(status="error", reason="torch_musa is required for MUSA replay")
            return result
        if not torch.musa.is_available():
            result.update(status="error", reason="No MUSA device is available")
            return result
    dtype_map = {name: getattr(torch, name) for name in (
        "float16", "float32", "float64", "bfloat16", "int8", "uint8", "int16", "int32", "int64", "bool")}
    def load_array(event):
        relative = event.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Invalid tensor file path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Tensor path escapes capture directory")
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("Replay tensor exceeds 256 MiB read limit")
        from .compare import _load_tensor
        return _load_tensor(root, event)
    try:
        values = {}
        for event in inputs:
            key = event["tensor_path"]
            if key in values:
                raise ValueError("Duplicate tensor path in API invocation")
            name = event["dtype"].removeprefix("torch.")
            if name not in dtype_map:
                raise ValueError(f"Unsupported logical dtype: {name}")
            tensor = torch.from_numpy(load_array(event).copy()).to(dtype=dtype_map[name])
            if reference_dtype == "float32" and tensor.is_floating_point():
                tensor = tensor.float()
            values[key] = tensor.to(device)
        def decode(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if not isinstance(value, dict):
                raise ValueError("Invalid call specification")
            if "tensor_path" in value:
                return values[value["tensor_path"]]
            if value.get("type") == "tuple":
                return tuple(decode(item) for item in value["items"])
            if value.get("type") == "list":
                return [decode(item) for item in value["items"]]
            if value.get("type") == "dict":
                return {key: decode(item) for key, item in value["items"]}
            raise ValueError("Call contains an unsupported scalar or object")
        args, kwargs = decode(spec["args"]), decode(spec["kwargs"])
        if not isinstance(args, (tuple, list)) or not isinstance(kwargs, dict):
            raise ValueError("API call arguments must be a tuple/list and dict")
        # Attribute access is only applied after an exact whitelist match, never from arbitrary input.
        _, packet, overload = op_name.split(".")
        function = getattr(getattr(torch.ops.aten, packet), overload)
        with torch.no_grad():
            actual = function(*args, **kwargs)
        if not isinstance(actual, torch.Tensor):
            raise ValueError("Replay output is not a tensor")
        expected_dtype = outputs[0]["dtype"]
        recorded_floating = expected_dtype in {"torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"}
        allowed_dtype = "torch.float32" if reference_dtype == "float32" and recorded_floating else expected_dtype
        if str(actual.dtype) != allowed_dtype:
            result.update(status="mismatch", reason="Replay output logical dtype differs",
                          expected_dtype=allowed_dtype, actual_dtype=str(actual.dtype))
            return result
        if actual.dtype == torch.bfloat16:
            actual = actual.float()
        actual = actual.detach().cpu().numpy()
        expected = load_array(outputs[0])
        if expected.shape != actual.shape:
            result.update(status="mismatch", reason="Replay output shape differs")
            return result
        dtype = outputs[0]["dtype"]
        defaults = (1e-2, 1e-2) if dtype == "torch.bfloat16" else ((1e-3, 1e-3) if dtype == "torch.float16" else (1e-5, 1e-4))
        atol = defaults[0] if atol is None else atol
        rtol = defaults[1] if rtol is None else rtol
        from .compare import _metrics
        metrics = _metrics(expected, actual, atol, rtol)
        finite = metrics["nonfinite_reference"] == 0 and metrics["nonfinite_candidate"] == 0
        same = metrics["mismatch_count"] == 0
        result.update(status="pass" if finite and same else "mismatch", finite=finite,
                      max_abs_error=metrics["max_abs_error"], metrics=metrics,
                      atol=atol, rtol=rtol, shape=list(expected.shape),
                      reason="Pure-op replay comparison of captured tensors; this does not restore model, optimizer or RNG state, or reconstruct a full parameter from gradient shards.")
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        result.update(status="unsupported", reason=f"Reference could not be evaluated: {type(exc).__name__}: {exc}")
    return result
