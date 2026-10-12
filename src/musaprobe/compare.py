"""Offline, NumPy-only inspection and comparison of musaProbe captures.

Input captures are never modified. Tensor files are memory mapped and processed
in bounded chunks; loading arbitrary Python objects is deliberately unsupported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_CHUNK_ELEMENTS = 262144
_KEY_FIELDS = ("step", "microbatch", "phase", "module", "call_index", "kind",
               "tensor_path", "op", "recompute", "parameter_name", "api_call_index",
               "config_version", "backward_id", "schema_version", "virtual_pipeline")
_REQUIRED_FIELDS = ("step", "phase", "module", "call_index", "kind", "tensor_path",
                    "shape", "dtype", "state")
_FAILURES = {"mismatch", "nonfinite", "shape_mismatch", "dtype_mismatch", "error", "unaligned"}
_DIFFERENCES = {"mismatch", "nonfinite", "shape_mismatch", "dtype_mismatch"}
_INCOMPLETE = {"missing_reference", "missing_candidate", "statistics_only", "truncated", "unsupported"}
_STATES = {"tensor", "statistics", "truncated", "unsupported", "error"}
_DTYPES = {"float16": "float16", "half": "float16", "float32": "float32", "float": "float32",
           "float64": "float64", "double": "float64", "bfloat16": "float32", "bool": "bool",
           "uint8": "uint8", "int8": "int8", "int16": "int16", "short": "int16",
           "int32": "int32", "int": "int32", "int64": "int64", "long": "int64",
           "uint16": "uint16", "uint32": "uint32", "uint64": "uint64"}
_DTYPE_ALIASES = {"half": "float16", "float": "float32", "double": "float64", "short": "int16",
                  "int": "int32", "long": "int64"}
_GRADIENT_LAYOUT_FIELDS = ("gradient_scope", "original_parameter_shape", "parameter_flat_range",
                           "gradient_stage", "gradient_parallel_coordinates")


def _dtype_name(value: str | None) -> str:
    if value is None:
        return "unknown"
    name = value.removeprefix("torch.").removeprefix("numpy.")
    return _DTYPE_ALIASES.get(name, name)


def _identity(event: dict[str, Any]) -> tuple[Any, ...]:
    effective = {**event, **event.get("_mapped_fields", {})}
    values = tuple(effective.get(k, 1 if k == "schema_version" else (False if k == "recompute" else None)) for k in _KEY_FIELDS)
    hash(values)  # Reject malformed container values instead of inventing an identity.
    return values


def _validate_slice(event: dict[str, Any]) -> None:
    if "tensor_slice" not in event:
        return
    spec = event["tensor_slice"]
    if not isinstance(spec, dict) or set(spec) != {"original_shape", "selection"}:
        raise ValueError("tensor_slice requires original_shape and selection")
    shape, selection = spec["original_shape"], spec["selection"]
    if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(selection, list) or len(shape) != len(selection)):
        raise ValueError("invalid tensor_slice dimensions")
    expected = []
    for size, axis in zip(shape, selection):
        if not isinstance(axis, dict) or set(axis) != {"start", "stop", "step"}:
            raise ValueError("slice axes require normalized start/stop/step")
        start, stop, step = (axis[k] for k in ("start", "stop", "step"))
        if any(type(x) is not int for x in (start, stop, step)) or step == 0:
            raise ValueError("slice indices must be integers with nonzero step")
        low, high = (0, size) if step > 0 else (-1, size - 1)
        if not low <= start <= high or not low <= stop <= high:
            raise ValueError("slice indices are outside the original shape")
        expected.append(len(range(start, stop, step)))
    if event["shape"] != expected:
        raise ValueError("tensor_slice selection does not match stored shape")


def _rank_number(path: Path) -> int:
    suffix = path.name.removeprefix("rank_")
    if not path.name.startswith("rank_") or not suffix.isdigit():
        raise ValueError(f"Invalid rank directory: {path.name}")
    return int(suffix)


def _validate_event(event: Any) -> None:
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    missing = [k for k in _REQUIRED_FIELDS if k not in event]
    if missing:
        raise ValueError("missing event fields: " + ", ".join(missing))
    if type(event.get("schema_version", 1)) is not int or event.get("schema_version", 1) not in (1, 2):
        raise ValueError("unsupported event schema_version")
    if event.get("schema_version", 1) == 2:
        if "config_version" not in event or "backward_id" not in event:
            raise ValueError("schema 2 event requires config_version and backward_id")
    if "config_version" in event and (type(event["config_version"]) is not int or event["config_version"] < 0):
        raise ValueError("config_version must be a nonnegative integer")
    if event.get("backward_id") is not None and type(event["backward_id"]) not in (str, int):
        raise ValueError("backward_id must be a string, integer or null")
    for name in ("step", "call_index"):
        if type(event[name]) is not int or event[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("microbatch", "api_call_index", "virtual_pipeline"):
        if event.get(name) is not None and (type(event[name]) is not int or event[name] < 0):
            raise ValueError(f"{name} must be a nonnegative integer or null")
    if "event_id" in event and (type(event["event_id"]) is not int or event["event_id"] < 0):
        raise ValueError("event_id must be a nonnegative integer")
    for name in ("phase", "module", "kind", "tensor_path"):
        if not isinstance(event[name], str):
            raise ValueError(f"{name} must be a string")
    for name in ("op", "parameter_name"):
        if event.get(name) is not None and not isinstance(event[name], str):
            raise ValueError(f"{name} must be a string or null")
    if "recompute" in event and type(event["recompute"]) is not bool:
        raise ValueError("recompute must be a boolean")
    if not isinstance(event["shape"], list) or any(type(n) is not int or n < 0 for n in event["shape"]):
        raise ValueError("shape must contain nonnegative integers")
    if event["state"] not in _STATES:
        raise ValueError("unknown event state")
    if not isinstance(event["dtype"], str) and not (event["dtype"] is None and event["state"] in {"truncated", "unsupported", "error"}):
        raise ValueError("dtype must be a string (or null for a non-tensor notice)")
    if "stats" in event and not isinstance(event["stats"], dict):
        raise ValueError("stats must be a JSON object")
    _validate_slice(event)
    _identity(event)


def _read_capture(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Capture directory does not exist: {root}")
    if (root / "events.jsonl").exists() or root.name.startswith("rank_"):
        rank_dirs = [root]
    else:
        rank_dirs = sorted(p for p in root.glob("rank_*") if p.is_dir())
    if not rank_dirs:
        raise ValueError(f"No rank capture directories found: {root}")
    result: dict[str, Any] = {"path": str(root), "ranks": {}, "errors": []}
    for rank_dir in rank_dirs:
        try:
            rank = _rank_number(rank_dir)
            if rank in result["ranks"]:
                raise ValueError(f"duplicate rank directory for rank {rank}")
        except ValueError as exc:
            result["errors"].append({"rank": None, "reason": str(exc)})
            continue
        entry: dict[str, Any] = {"dir": rank_dir, "events": [], "manifest": {}, "errors": []}
        result["ranks"][rank] = entry
        try:
            with (rank_dir / "manifest.json").open(encoding="utf-8") as stream:
                entry["manifest"] = json.load(stream)
            if not isinstance(entry["manifest"], dict):
                raise ValueError("manifest must be a JSON object")
            if type(entry["manifest"].get("schema_version", 1)) is not int or entry["manifest"].get("schema_version", 1) not in (1, 2):
                raise ValueError("unsupported manifest schema_version")
            if "rank" in entry["manifest"] and entry["manifest"]["rank"] != rank:
                raise ValueError("manifest rank differs from directory rank")
        except (OSError, ValueError, TypeError) as exc:
            entry["errors"].append(f"manifest: {exc}")
            entry["manifest"] = {}
        try:
            with (rank_dir / "events.jsonl").open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        _validate_event(event)
                        event = dict(event)
                        event["_line_number"] = line_number
                        entry["events"].append(event)
                    except (ValueError, TypeError) as exc:
                        entry["errors"].append(f"events.jsonl line {line_number}: {exc}")
        except (OSError, UnicodeError) as exc:
            entry["errors"].append(f"events.jsonl: {exc}")
        manifest = entry["manifest"]
        if manifest.get("status") == "complete" and "events_count" in manifest:
            count = manifest["events_count"]
            if type(count) is not int or count != len(entry["events"]):
                entry["errors"].append("complete manifest events_count differs from readable event count")
        result["errors"].extend({"rank": rank, "reason": reason} for reason in entry["errors"])
    return result


def _nonfinite_counts(event: dict[str, Any]) -> dict[str, int]:
    stats = event.get("stats", {})
    result = {}
    for key in ("nan_count", "posinf_count", "neginf_count"):
        value = stats.get(key, 0)
        result[key] = value if type(value) is int and value >= 0 else 0
    return result


def _capture_summary(capture: dict[str, Any]) -> dict[str, Any]:
    states: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    nonfinite: list[dict[str, Any]] = []
    duplicate_identities: list[dict[str, Any]] = []
    rank_summaries = []
    for rank, entry in sorted(capture["ranks"].items()):
        events = entry["events"]
        config = entry["manifest"].get("config") or {}
        rank_filter = config.get("ranks") if isinstance(config, dict) else None
        keys: Counter[tuple[Any, ...]] = Counter(_identity(event) for event in events)
        for key, count in keys.items():
            if count > 1:
                duplicate_identities.append({"rank": rank, "identity": dict(zip(_KEY_FIELDS, key)), "count": count})
        for event in events:
            states[event["state"]] += 1
            kinds[event["kind"]] += 1
            counts = _nonfinite_counts(event)
            if any(counts.values()):
                nonfinite.append({"rank": rank, "identity": dict(zip(_KEY_FIELDS, _identity(event))), **counts})
        rank_summaries.append({
            "rank": rank, "events": len(events), "status": entry["manifest"].get("status", "unknown"),
            "configured_rank_filter": rank_filter,
            "excluded_by_rank_filter": isinstance(rank_filter, list) and rank not in rank_filter,
            "steps": sorted({event["step"] for event in events}),
            "microbatches": sorted({event["microbatch"] for event in events if event.get("microbatch") is not None}),
            "modules": sorted({event["module"] for event in events}),
            "errors": entry["errors"],
        })
    return {"path": capture["path"], "ranks": rank_summaries,
            "recorded_ranks": [item["rank"] for item in rank_summaries if item["events"]],
            "events": sum(states.values()),
            "states": dict(states), "kinds": dict(kinds), "nonfinite": nonfinite,
            "duplicate_identities": duplicate_identities, "errors": capture["errors"],
            "note": "Coverage describes recorded events only; statistics are not complete tensor verification."}


def inspect_run(path: str | Path) -> dict[str, Any]:
    """Summarize capture metadata, coverage, and recorded NaN/Inf observations.

    This inexpensive inspection does not scan tensor payloads. Use compare_runs
    for payload validation. Interrupted captures retain their completed events.
    """
    return {"schema_version": 1, **_capture_summary(_read_capture(path))}


def _load_tensor(rank_dir: Path, event: dict[str, Any]) -> np.ndarray:
    if event.get("fp8") is not None:
        from .fp8_numpy import FP8Tensor
        return FP8Tensor(rank_dir, event)
    value = event.get("file")
    if not isinstance(value, str) or not value:
        raise ValueError("tensor event has no relative tensor file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe tensor file reference: absolute paths and traversal are forbidden")
    target = (rank_dir / relative).resolve()
    if not target.is_relative_to(rank_dir.resolve()):
        raise ValueError("unsafe tensor file reference: symlink escapes the rank directory")
    if target.suffix != ".npy":
        raise ValueError("only .npy tensor payloads are supported")
    array = np.load(target, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("payload is not an ndarray")
    if list(array.shape) != event["shape"]:
        raise ValueError("payload shape differs from event metadata")
    dtype_name = _dtype_name(event["dtype"])
    expected = _DTYPES.get(dtype_name)
    if expected is None:
        raise TypeError(f"unsupported logical dtype: {event['dtype']}")
    if array.dtype != np.dtype(expected):
        raise ValueError(f"payload dtype {array.dtype} differs from logical dtype {event['dtype']}")
    numel = event.get("stats", {}).get("numel")
    if numel is not None and numel != array.size:
        raise ValueError("payload element count differs from event metadata")
    return array


def _finite_float(value: Any) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        number = float(value)
    return number if math.isfinite(number) else None


def _metrics(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float, *,
             compute_ulp: bool = False, max_ulp: int | None = None) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError("metric inputs must have equal shapes")
    ulp_applicable = compute_ulp and reference.dtype == np.dtype("float32") and candidate.dtype == np.dtype("float32")
    if max_ulp is not None and not ulp_applicable and reference.dtype.kind not in "biu":
        raise TypeError("max_ulp is supported only for logical FP32 tensors")
    count = reference.size
    exact = reference.dtype.kind in "biu"
    mismatch_count = 0
    nonfinite_reference = 0
    nonfinite_candidate = 0
    finite_pairs = 0
    max_abs = np.longdouble(0)
    max_rel = np.longdouble(0)
    absolute_sum = np.longdouble(0)
    ref_scale = np.longdouble(0)
    cand_scale = np.longdouble(0)
    diff_scale = np.longdouble(0)
    ref_ss = np.longdouble(0)
    cand_ss = np.longdouble(0)
    diff_ss = np.longdouble(0)
    cross = np.longdouble(0)
    first_bad = None
    max_ulp_error, ulp_mismatch_count, tolerance_mismatch_count = 0, 0, 0
    for start in range(0, count, _CHUNK_ELEMENTS):
        # flat slices preserve logical C order even when NPY is Fortran ordered.
        original_ref = reference.flat[start:start + _CHUNK_ELEMENTS]
        original_cand = candidate.flat[start:start + _CHUNK_ELEMENTS]
        ref = original_ref.astype(np.longdouble)
        cand = original_cand.astype(np.longdouble)
        ref_finite, cand_finite = np.isfinite(ref), np.isfinite(cand)
        finite = ref_finite & cand_finite
        nonfinite_reference += int(np.count_nonzero(~ref_finite))
        nonfinite_candidate += int(np.count_nonzero(~cand_finite))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            diff = np.abs(cand - ref)
            bad = original_ref != original_cand if exact else diff > atol + rtol * np.abs(ref)
            bad = bad | ~finite
        tolerance_mismatch_count += int(np.count_nonzero(bad))
        if ulp_applicable and np.any(finite):
            # Both signed zeros map to one point; adjacent finite FP32 values
            # have consecutive keys, including the negative/positive boundary.
            keys = []
            for values in (original_ref[finite], original_cand[finite]):
                bits = values.view(np.uint32).astype(np.int64)
                magnitude = bits & 0x7fffffff
                keys.append(np.where(bits & 0x80000000, 0x80000000 - magnitude, 0x80000000 + magnitude))
            distance = np.abs(keys[1] - keys[0])
            max_ulp_error = max(max_ulp_error, int(np.max(distance)))
            if max_ulp is not None:
                ulp_bad = distance > max_ulp
                ulp_mismatch_count += int(np.count_nonzero(ulp_bad))
                bad[finite] |= ulp_bad
        mismatch_count += int(np.count_nonzero(bad))
        if first_bad is None and np.any(bad):
            first_bad = start + int(np.flatnonzero(bad)[0])
        if not np.any(finite):
            continue
        ref, cand, diff = ref[finite], cand[finite], diff[finite]
        finite_pairs += ref.size
        absolute_sum += np.sum(diff, dtype=np.longdouble)
        max_abs = max(max_abs, np.max(diff))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            relative = np.where(ref != 0, diff / np.abs(ref), np.where(diff == 0, 0, np.inf))
        max_rel = max(max_rel, np.max(relative))
        new_ref_scale = max(ref_scale, np.max(np.abs(ref)))
        new_cand_scale = max(cand_scale, np.max(np.abs(cand)))
        new_diff_scale = max(diff_scale, np.max(diff))
        ref_ratio = ref_scale / new_ref_scale if new_ref_scale else np.longdouble(0)
        cand_ratio = cand_scale / new_cand_scale if new_cand_scale else np.longdouble(0)
        diff_ratio = diff_scale / new_diff_scale if new_diff_scale else np.longdouble(0)
        ref_scaled = ref / new_ref_scale if new_ref_scale else np.zeros_like(ref)
        cand_scaled = cand / new_cand_scale if new_cand_scale else np.zeros_like(cand)
        ref_ss = ref_ss * ref_ratio ** 2 + np.sum(ref_scaled ** 2, dtype=np.longdouble)
        cand_ss = cand_ss * cand_ratio ** 2 + np.sum(cand_scaled ** 2, dtype=np.longdouble)
        cross = cross * ref_ratio * cand_ratio + np.sum(ref_scaled * cand_scaled, dtype=np.longdouble)
        if new_diff_scale:
            diff_ss = diff_ss * diff_ratio ** 2 + np.sum((diff / new_diff_scale) ** 2, dtype=np.longdouble)
        ref_scale, cand_scale, diff_scale = new_ref_scale, new_cand_scale, new_diff_scale
    if finite_pairs:
        ref_norm = ref_scale * np.sqrt(ref_ss)
        diff_norm = diff_scale * np.sqrt(diff_ss)
        relative_l2 = diff_norm / ref_norm if ref_norm else (0 if not diff_norm else np.inf)
        if ref_ss and cand_ss:
            cosine = np.clip(cross / np.sqrt(ref_ss * cand_ss), -1, 1)
        else:
            cosine = 1.0 if not ref_ss and not cand_ss else None
    else:
        relative_l2, cosine = (0.0, 1.0) if count == 0 else (None, None)
    return {"numel": count, "finite_pairs": finite_pairs, "mismatch_count": mismatch_count,
            "mismatch_ratio": mismatch_count / count if count else 0.0,
            "max_abs_error": _finite_float(max_abs) if finite_pairs or not count else None,
            "mean_abs_error": _finite_float(absolute_sum / finite_pairs) if finite_pairs else (0.0 if not count else None),
            "max_relative_error": _finite_float(max_rel) if finite_pairs or not count else None,
            "relative_error_unbounded": bool(np.isinf(max_rel)),
            "cosine_similarity": _finite_float(cosine) if cosine is not None else None,
            "relative_l2": _finite_float(relative_l2) if relative_l2 is not None else None,
            "relative_l2_unbounded": bool(relative_l2 is not None and np.isinf(relative_l2)),
            "nonfinite_reference": nonfinite_reference, "nonfinite_candidate": nonfinite_candidate,
            "first_mismatch_flat_index": first_bad, "atol": 0.0 if exact else atol,
            "rtol": 0.0 if exact else rtol, "exact_comparison": exact,
            "ulp_applicable": ulp_applicable,
            "max_ulp_error": max_ulp_error if ulp_applicable and (finite_pairs or not count) else None,
            "max_ulp_threshold": max_ulp if ulp_applicable else None,
            "ulp_mismatch_count": ulp_mismatch_count if ulp_applicable else None,
            "tolerance_mismatch_count": tolerance_mismatch_count}


def _public_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    return {key: value for key, value in event.items() if not key.startswith("_")} if event else None


def _tolerances(dtype, atol=None, rtol=None):
    defaults = {"float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2)}.get(_dtype_name(dtype), (1e-5, 1e-4))
    return defaults[0] if atol is None else atol, defaults[1] if rtol is None else rtol


def _compare_pair(record: dict[str, Any], ref: dict[str, Any], cand: dict[str, Any],
                  ref_dir: Path, cand_dir: Path, atol: float | None, rtol: float | None,
                  max_ulp: int | None = None) -> None:
    for key in ("optimizer_type", "state_stage", "reconstruction_coordinates", "reconstruction_scope"):
        if (key in ref) != (key in cand) or ref.get(key) != cand.get(key):
            record.update(status="unaligned", reason=f"{key} differs or is missing on one side.")
            return
    if ref.get("fp8") is not None or cand.get("fp8") is not None:
        fields = ("format", "encoding", "logical_shape", "scale_mode", "block_m", "block_n")
        layouts = [{key: event.get("fp8", {}).get(key) for key in fields} if isinstance(event.get("fp8"), dict) else None for event in (ref, cand)]
        record["fp8_quantization"] = {"reference": ref.get("fp8"), "candidate": cand.get("fp8"),
                                      "metric_space": "decoded_float32", "training_state_restored": False}
        if layouts[0] != layouts[1]:
            record.update(status="unaligned", reason="FP8 format or scale layout differs or is absent on one side.")
            return
    if ref.get("tensor_slice") != cand.get("tensor_slice"):
        record.update(status="unaligned", reason="Tensor slice layout differs; equal stored shapes do not identify the same elements.")
        return
    if any(key in event for key in _GRADIENT_LAYOUT_FIELDS for event in (ref, cand)):
        layouts = {name: {key: event[key] for key in _GRADIENT_LAYOUT_FIELDS if key in event}
                   for name, event in (("reference", ref), ("candidate", cand))}
        record["gradient_layout"] = layouts
        problems = []
        for key in _GRADIENT_LAYOUT_FIELDS:
            if (key in ref) != (key in cand):
                problems.append(f"{key} is present on only one side")
            elif key in ref and ref[key] != cand[key]:
                problems.append(f"{key} differs")
        for name, event in (("reference", ref), ("candidate", cand)):
            if event.get("gradient_scope") == "optimizer_owned_shard":
                missing = [key for key in _GRADIENT_LAYOUT_FIELDS[1:] if event.get(key) is None]
                if missing:
                    problems.append(f"{name} optimizer shard lacks " + ", ".join(missing))
        if problems:
            record.update(status="unaligned", reason="Gradient layout/stage cannot be aligned: " + "; ".join(problems)
                          + ". Equal flattened shapes do not establish equal parameter slices.")
            return
    for state in ("error", "unsupported", "truncated"):
        if ref["state"] == state or cand["state"] == state:
            record.update(status=state, reason="; ".join(f"{name}: {event.get('reason') or event['state']}"
                          for name, event in (("reference", ref), ("candidate", cand)) if event["state"] == state))
            return
    if ref["shape"] != cand["shape"]:
        record.update(status="shape_mismatch", reason="Tensor shapes differ.")
        return
    if _dtype_name(ref["dtype"]) != _dtype_name(cand["dtype"]):
        record.update(status="dtype_mismatch", reason="Logical tensor dtypes differ.")
        return
    if ref["state"] != "tensor" or cand["state"] != "tensor":
        if any(_nonfinite_counts(ref).values()) or any(_nonfinite_counts(cand).values()):
            record.update(status="nonfinite", reason="Recorded statistics contain NaN or Inf; full tensors were not compared.")
        else:
            record.update(status="statistics_only", reason="Full tensor comparison requires tensor payloads on both sides.")
        return
    try:
        reference = _load_tensor(ref_dir, ref)
        candidate = _load_tensor(cand_dir, cand)
        name = _dtype_name(ref["dtype"])
        effective_atol, effective_rtol = _tolerances(name, atol, rtol)
        if ref.get("fp8") is not None and max_ulp is not None:
            raise TypeError("FP32 ULP thresholds do not apply to decoded FP8 quantized values")
        metrics = _metrics(reference, candidate, effective_atol, effective_rtol,
                           compute_ulp=name == "float32" and ref.get("fp8") is None, max_ulp=max_ulp)
        record["metrics"] = metrics
        if metrics["nonfinite_reference"] or metrics["nonfinite_candidate"]:
            record.update(status="nonfinite", reason="At least one full tensor contains NaN or Inf.")
        elif metrics["mismatch_count"]:
            record.update(status="mismatch", reason="Tensor elements exceed an enabled comparison threshold.")
        else:
            record.update(status="pass", reason="All tensor elements satisfy the comparison rule.")
    except TypeError as exc:
        record.update(status="unsupported", reason=str(exc))
    except (OSError, ValueError, EOFError, OverflowError) as exc:
        record.update(status="error", reason=f"Tensor payload could not be verified: {exc}")


def _sort_record(record: dict[str, Any]) -> tuple[Any, ...]:
    event = record.get("candidate") or record.get("reference") or {}
    return (event.get("step", -1), record["rank"] if record["rank"] is not None else -1,
            event.get("_line_number", event.get("event_id", -1)),
            json.dumps(record.get("identity", {}), sort_keys=True))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_outputs(comparison: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    columns = ["rank", "status", *_KEY_FIELDS, "reason", "numel", "mismatch_count", "mismatch_ratio",
               "max_abs_error", "mean_abs_error", "max_relative_error", "cosine_similarity", "relative_l2",
               "max_ulp_error", "max_ulp_threshold", "ulp_mismatch_count"]
    with (output_path / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in comparison["records"]:
            row = {**record, **record.get("identity", {}), **record.get("metrics", {})}
            # Prevent spreadsheet formula execution for untrusted module names/stacks.
            for key, value in row.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    row[key] = "'" + value
            writer.writerow(row)
    from .report import render_report
    render_report(comparison, output_path / "report.html")


def compare_runs(reference: str | Path, candidate: str | Path, *, atol: float | None = None,
                 rtol: float | None = None, max_ulp: int | None = None, rules=None, mapping=None,
                 output: str | Path | None = None) -> dict[str, Any]:
    """Compare semantically matching tensors; incomplete data can never pass.

    Default float tolerances depend on the logical dtype. Integer and boolean
    tensors always compare exactly. Nonfinite values fail even if equal.
    """
    from .policies import ThresholdRules, ExplicitMapping, validate_tolerance
    for name, value in (("atol", atol), ("rtol", rtol), ("max_ulp", max_ulp)):
        validate_tolerance(name, value)
    policy, mapper = ThresholdRules(rules), ExplicitMapping(mapping)
    ref_capture, cand_capture = _read_capture(reference), _read_capture(candidate)
    records = []
    for side, capture in (("reference", ref_capture), ("candidate", cand_capture)):
        for issue in capture["errors"]:
            records.append({"rank": issue["rank"], "identity": {}, "status": "error",
                            "reason": f"{side}: {issue['reason']}", "reference": None, "candidate": None, "metrics": {}})
    all_ranks = sorted(set(ref_capture["ranks"]) | set(cand_capture["ranks"]))
    for rank in all_ranks:
        ref_entry, cand_entry = ref_capture["ranks"].get(rank), cand_capture["ranks"].get(rank)
        if ref_entry and cand_entry:
            ref_manifest, cand_manifest = ref_entry["manifest"], cand_entry["manifest"]
            layouts = []
            for manifest in (ref_manifest, cand_manifest):
                metadata = manifest.get("metadata") or {}
                layouts.append({"world_size": manifest.get("world_size"),
                                "parallelism": metadata.get("parallelism") if isinstance(metadata, dict) else None})
            if layouts[0] != layouts[1]:
                records.append({"rank": rank, "identity": {}, "status": "unaligned", "reason": "Parallel layout metadata differs; rank tensors were not compared.",
                                "reference": None, "candidate": None, "metrics": {}, "layouts": layouts})
                continue
        elif not (ref_entry or cand_entry)["events"]:
            records.append({"rank": rank, "identity": {}, "status": "missing_reference" if ref_entry is None else "missing_candidate",
                            "reason": "An empty rank capture is missing from the other run.", "reference": None, "candidate": None, "metrics": {}})
        ref_events: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        cand_events: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for side, entry, index in (("reference", ref_entry, ref_events), ("candidate", cand_entry, cand_events)):
            for event in entry["events"] if entry else []:
                event = mapper.apply(event, rank, side)
                index[_identity(event)].append(event)
        for key in dict.fromkeys([*ref_events, *cand_events]):
            refs, cands = ref_events.get(key, []), cand_events.get(key, [])
            ref, cand = refs[0] if refs else None, cands[0] if cands else None
            record = {"rank": rank, "identity": dict(zip(_KEY_FIELDS, key)), "status": None,
                      "reason": None, "reference": _public_event(ref), "candidate": _public_event(cand), "metrics": {}}
            if any(event and event.get("_mapping_id") for event in (ref, cand)):
                record["mapping"] = {side: {"rule": event.get("_mapping_id"),
                    "original_identity": {k: event.get(k) for k in _KEY_FIELDS}}
                    for side, event in (("reference", ref), ("candidate", cand)) if event}
            if len(refs) > 1 or len(cands) > 1:
                record.update(status="unaligned", reason=f"Duplicate semantic identity: reference={len(refs)}, candidate={len(cands)}; no event was selected for comparison.",
                              duplicate_reference_count=len(refs), duplicate_candidate_count=len(cands))
            elif ref is None:
                record.update(status="missing_reference", reason="No reference event has the same rank and semantic identity.")
            elif cand is None:
                record.update(status="missing_candidate", reason="No candidate event has the same rank and semantic identity.")
            elif ref.get("_mapping_id") != cand.get("_mapping_id"):
                record.update(status="unaligned", reason="An explicit mapping must select both sides of the same one-to-one pair.")
            else:
                thresholds, evidence = policy.resolve({**ref, **ref.get("_mapped_fields", {})}, atol=atol, rtol=rtol, max_ulp=max_ulp)
                record["threshold_policy"] = evidence
                _compare_pair(record, ref, cand, ref_entry["dir"], cand_entry["dir"],
                              thresholds.get("atol"), thresholds.get("rtol"), thresholds.get("max_ulp"))
            records.append(record)
    for pair in mapper.pairs:
        absent = [side for side in ("reference", "candidate") if not mapper.uses[side].get(pair["id"])]
        if absent:
            records.append({"rank": None, "identity": {}, "status": "unaligned", "reference": None,
                            "candidate": None, "metrics": {}, "reason":
                            f"Mapping {pair['id']} selected no events on: {', '.join(absent)}"})
    records.sort(key=_sort_record)
    counts = Counter(record["status"] for record in records)
    ref_coverage, cand_coverage = _capture_summary(ref_capture), _capture_summary(cand_capture)
    incomplete_runs = any(rank["status"] != "complete" for coverage in (ref_coverage, cand_coverage) for rank in coverage["ranks"])
    if any(counts[status] for status in _FAILURES):
        overall = "fail"
    elif not records or incomplete_runs or any(counts[status] for status in _INCOMPLETE):
        overall = "incomplete"
    else:
        overall = "pass"
    first_difference = next((record for record in records if record["status"] in _DIFFERENCES), None)
    comparison = _json_safe({"schema_version": 1, "name": "musaProbe", "reference": ref_capture["path"],
                            "candidate": cand_capture["path"], "overall": overall,
                            "summary": {"records": len(records), "statuses": dict(counts), "verified_tensors": counts["pass"] + counts["mismatch"] + sum(bool(r["metrics"]) for r in records if r["status"] == "nonfinite"),
                                        "incomplete_runs": incomplete_runs},
                            "coverage": {"reference": ref_coverage, "candidate": cand_coverage},
                            "first_difference": first_difference, "records": records,
                            "notes": ["First difference is an observed discrepancy, not a proven root cause.",
                                      "Ranks are compared by rank number; different parallel layouts are not reconstructed.",
                                      "Finite metrics exclude NaN/Inf pairs; any nonfinite value is a failure.",
                                      "A null metric denotes an undefined or unrepresentable value, never zero."]})
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        for capture in (ref_capture, cand_capture):
            source_path = Path(capture["path"])
            if output_path == source_path or output_path.is_relative_to(source_path):
                raise ValueError("comparison output must be outside input capture directories")
        _write_outputs(comparison, output)
    return comparison
