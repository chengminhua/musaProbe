"""Explicit rectangular tensor reconstruction; never infer Megatron name rules."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .analysis import protect_output, write_json
from .compare import _dtype_name, _load_tensor, _public_event, _read_capture
from .layout import validate_shape
from .policies import load_object


_CONTEXT = ("step", "microbatch", "phase", "kind", "tensor_path", "op", "recompute",
            "config_version", "backward_id", "schema_version", "gradient_stage", "optimizer_type", "state_stage")


def _context(event):
    return {key: event.get(key, 1 if key == "schema_version" else (False if key == "recompute" else None)) for key in _CONTEXT}


def _coverage(shape, boxes):
    if len(boxes) > 4096:
        raise ValueError("more than 4096 shards per tensor exceeds layout validation budget")
    seen, volume = [], 0
    for offset, size in boxes:
        validate_shape(list(offset))
        validate_shape(list(size))
        if len(offset) != len(shape) or len(size) != len(shape) or any(a + b > n for a, b, n in zip(offset, size, shape)):
            raise ValueError("shard rectangle is outside global_shape")
        if (offset, size) in seen:
            raise ValueError("duplicate shard rectangle")
        if any(all(max(a, c) < min(a + b, c + d) for a, b, c, d in zip(offset, size, other, other_size)) for other, other_size in seen):
            raise ValueError("shard rectangles overlap")
        seen.append((offset, size))
        volume += math.prod(size)
    if volume != math.prod(shape):
        raise ValueError("shard rectangles have a coverage gap")


def reconstruct_axes(capture, layout, output, *, max_bytes=1073741824):
    """Reconstruct exactly declared global tensors from rank/event rectangles.

    Layout: {schema_version:1,groups:[{name,global_shape,shards:[
    {rank,event_id,offsets}]}]}. Each shard's shape comes from its saved tensor.
    Module names and virtual pipeline coordinates are preserved as source
    evidence; the explicit global group name is the normalized output identity.
    This verifies physical coverage, not the user's semantic naming decision.
    """
    protect_output(output, (capture, layout))
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise ValueError("axis reconstruction requires a new or empty output directory")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    declaration = load_object(layout, name="axis layout")
    if set(declaration) - {"schema_version", "groups"} or declaration.get("schema_version", 1) != 1 or not isinstance(declaration.get("groups"), list):
        raise ValueError("axis layout requires schema_version 1 and groups")
    names = set()
    for group in declaration["groups"]:
        if (not isinstance(group, dict) or set(group) != {"name", "global_shape", "shards"}
                or not isinstance(group.get("name"), str) or not group["name"] or group["name"] in names):
            raise ValueError("axis groups require unique name, global_shape and shards")
        names.add(group["name"])
        validate_shape(group["global_shape"])
        if not isinstance(group["shards"], list) or not group["shards"]:
            raise ValueError("axis group requires explicit shards")
        for shard in group["shards"]:
            if not isinstance(shard, dict) or set(shard) != {"rank", "event_id", "offsets"}:
                raise ValueError("axis shard requires rank, event_id and offsets")
            if any(type(shard[k]) is not int or shard[k] < 0 for k in ("rank", "event_id")):
                raise ValueError("axis shard rank/event_id must be nonnegative integers")
            validate_shape(shard["offsets"])
    source = _read_capture(capture)
    lookup = defaultdict(list)
    for rank, entry in source["ranks"].items():
        for event in entry["events"]:
            if event.get("event_id") is not None:
                lookup[(rank, event["event_id"])].append((entry, event))
    used, records, prepared, total = set(), [], [], 0
    for group in declaration["groups"]:
        record = {"name": group["name"], "global_shape": group["global_shape"], "status": "incomplete", "sources": []}
        try:
            sources, boxes, first = [], [], None
            local_used = set()
            for shard in group["shards"]:
                key = (shard["rank"], shard["event_id"])
                if key in used or key in local_used:
                    raise ValueError("one captured event cannot be reused as multiple declared shards")
                local_used.add(key)
                choices = lookup[key]
                if len(choices) != 1:
                    raise ValueError("shard event is missing or event_id is ambiguous")
                entry, event = choices[0]
                if entry["errors"] or entry["manifest"].get("status") != "complete" or event["state"] != "tensor":
                    raise ValueError("source shard is not a complete tensor capture")
                if event.get("tensor_slice") is not None or event.get("fp8") is not None:
                    raise ValueError("sliced/encoded FP8 tensors need a separate explicit transformation")
                if first is None:
                    first = event
                elif _context(first) != _context(event) or _dtype_name(first["dtype"]) != _dtype_name(event["dtype"]):
                    raise ValueError("shards disagree on execution context, stage or logical dtype")
                array = _load_tensor(entry["dir"], event)
                offset = tuple(shard["offsets"])
                boxes.append((offset, array.shape))
                sources.append((offset, array))
                record["sources"].append({**shard, "shape": list(array.shape), "module": event["module"],
                                          "parameter_name": event.get("parameter_name"), "virtual_pipeline": event.get("virtual_pipeline"),
                                          "gradient_parallel_coordinates": event.get("gradient_parallel_coordinates")})
            _coverage(tuple(group["global_shape"]), boxes)
            bytes_needed = math.prod(group["global_shape"]) * sources[0][1].dtype.itemsize
            if total + bytes_needed > max_bytes:
                raise ValueError("axis reconstruction exceeds max_bytes budget")
            total += bytes_needed
            used.update(local_used)
            record.update(status="complete", bytes=bytes_needed, dtype=first["dtype"], context=_context(first))
            prepared.append((group, record, first, sources))
        except (ValueError, TypeError, OSError, KeyError) as exc:
            record["reason"] = str(exc)
        records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "rank_00000"
    directory.mkdir()
    events = []
    for group, record, first, sources in prepared:
        filename = f"tensor_{len(events):08d}.npy"
        destination = np.lib.format.open_memmap(directory / filename, mode="w+", dtype=sources[0][1].dtype, shape=tuple(group["global_shape"]))
        for offset, array in sources:
            selection = tuple(slice(a, a + size) for a, size in zip(offset, array.shape))
            # Both arrays are mapped and cannot alias: output lies outside inputs.
            np.copyto(destination[selection] if selection else destination, array, casting="no")
        destination.flush()
        del destination
        event = _public_event(first)
        for key in ("parameter_flat_range", "gradient_parallel_coordinates", "stride", "device", "file", "stats", "reconstruction_coordinates"):
            event.pop(key, None)
        event.update(event_id=len(events), module=group["name"], parameter_name=group["name"], call_index=0,
                     api_call_index=None, virtual_pipeline=None, shape=group["global_shape"], file=filename,
                     stats={"numel": math.prod(group["global_shape"])}, axis_reconstruction=record["sources"],
                     reconstruction_scope="explicit_global_axes_v1")
        if "gradient_scope" in event:
            event.update(gradient_scope="full_parameter", original_parameter_shape=group["global_shape"])
        events.append(event)
    complete = bool(records) and not source["errors"] and all(record["status"] == "complete" for record in records)
    status = "complete" if complete else "incomplete"
    (directory / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in events), encoding="utf-8")
    write_json(directory / "manifest.json", {"schema_version": 2, "rank": 0, "world_size": 1, "status": status,
        "events_count": len(events), "config": {}, "metadata": {"parallelism": {"layout": "explicit_global_axes_v1"},
            "axis_layout": declaration, "source": str(Path(capture).resolve())}})
    result = {"analysis_schema_version": 1, "kind": "axis_reconstruction", "status": status, "records": records,
              "bytes": total, "errors": source["errors"], "output": str(output.resolve()),
              "scope": "Only explicitly declared rectangles and global names. Exact coverage verified; no inferred TP/PP/VPP parameter naming or real-model VPP validation."}
    write_json(output / "axis_reconstruction.json", result)
    return result
