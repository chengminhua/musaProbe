"""Validated contiguous flat shards. No tensor-parallel axis inference."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .analysis import protect_output, write_json
from .compare import _KEY_FIELDS, _CHUNK_ELEMENTS, _dtype_name, _load_tensor, _read_capture


def validate_shape(shape):
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("original_shape must contain nonnegative integers")
    return tuple(shape)


class ContiguousTensor:
    """Array-like bounded reader used by the common metric engine.

    Each segment is (start, stop, ndarray). Exact, non-overlapping coverage is
    mandatory before any data can be read. This never allocates a full tensor.
    """
    def __init__(self, shape, segments):
        self.shape = validate_shape(list(shape))
        self.size = math.prod(self.shape)
        if not segments:
            raise ValueError("no tensor shards")
        for start, stop, array in segments:
            if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.size:
                raise ValueError("invalid shard half-open flat range")
            if stop - start != array.size:
                raise ValueError("shard range length differs from stored element count")
        self.segments = sorted(segments, key=lambda s: (s[0], s[1]))
        self.dtype = self.segments[0][2].dtype
        cursor = 0
        empty_count = 0
        for start, stop, array in self.segments:
            if array.dtype != self.dtype:
                raise ValueError("shard payload dtypes differ")
            if start == stop:
                empty_count += 1
                if self.size or empty_count > 1:
                    raise ValueError("redundant or empty shard range")
            if start != cursor:
                raise ValueError("shard ranges overlap or contain a gap")
            cursor = stop
        if cursor != self.size:
            raise ValueError("shard coverage is incomplete")
        self.flat = self

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("contiguous flat slices only")
        start, stop, _ = selection.indices(self.size)
        result = np.empty(max(0, stop - start), dtype=self.dtype)
        for a, b, array in self.segments:
            lo, hi = max(a, start), min(b, stop)
            if hi > lo:
                result[lo - start:hi - start] = array.flat[lo - a:hi - a]
        return result

    def save(self, path):
        target = np.lib.format.open_memmap(path, mode="w+", dtype=self.dtype, shape=self.shape)
        for start in range(0, self.size, _CHUNK_ELEMENTS):
            stop = min(start + _CHUNK_ELEMENTS, self.size)
            target.flat[start:stop] = self[start:stop]
        target.flush()
        del target


def reconstruct_shards(capture, output, *, parameter=None, step=None, max_bytes=1073741824):
    """Rebuild complete DP optimizer-owned flat parameter shards only.

    TP/PP/virtual-pipeline coordinates remain separate. Their tensors are not
    claimed to be the full model-global parameter. Inputs are never modified.
    """
    protect_output(output, (capture,))
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise ValueError("reconstruction requires a new or empty output directory")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("step must be nonnegative")
    source = _read_capture(capture)
    groups, records, prepared = defaultdict(list), [], []
    fields = tuple(k for k in _KEY_FIELDS if k != "schema_version")
    for rank, entry in source["ranks"].items():
        for event in entry["events"]:
            if event.get("gradient_scope") not in ("optimizer_owned_shard", "not_owned_by_rank"):
                continue
            if parameter is not None and event.get("parameter_name") != parameter:
                continue
            if step is not None and event["step"] != step:
                continue
            # Local call indices must be stable across ranks; do not guess names.
            identity = {k: event.get(k, False if k == "recompute" else None) for k in fields}
            coords = event.get("gradient_parallel_coordinates")
            topology = {k: v for k, v in coords.items() if k != "data"} if isinstance(coords, dict) else None
            key = json.dumps({"identity": identity, "coordinates": topology}, sort_keys=True)
            groups[key].append((rank, entry, event))
    total_bytes = 0
    for key, members in groups.items():
        record = {**json.loads(key), "status": "incomplete", "sources": [], "reason": ""}
        try:
            owners = [(rank, entry, event) for rank, entry, event in members if event["gradient_scope"] == "optimizer_owned_shard"]
            if not owners:
                raise ValueError("all observed ranks explicitly do not own this parameter")
            first = owners[0][2]
            shape = validate_shape(first.get("original_parameter_shape"))
            dtype, stage = first["dtype"], first.get("gradient_stage")
            if not isinstance(stage, str) or not stage:
                raise ValueError("gradient_stage is required")
            coords_seen, segments = set(), []
            for rank, entry, event in owners:
                coords = event.get("gradient_parallel_coordinates")
                if not isinstance(coords, dict) or not {"tensor", "pipeline", "data"}.issubset(coords):
                    raise ValueError("explicit tensor/pipeline/data coordinates are required")
                if any(type(v) is not int or v < 0 for v in coords.values()):
                    raise ValueError("parallel coordinates must be nonnegative integers")
                encoded = json.dumps(coords, sort_keys=True)
                if encoded in coords_seen:
                    raise ValueError("duplicate ownership for the same parallel coordinates")
                coords_seen.add(encoded)
                if (event.get("original_parameter_shape") != list(shape) or _dtype_name(event["dtype"]) != _dtype_name(dtype)
                        or event.get("gradient_stage") != stage or event.get("schema_version", 1) != first.get("schema_version", 1)):
                    raise ValueError("shard shape/dtype/stage/schema metadata differs")
                if event.get("tensor_slice") is not None or event.get("fp8") is not None:
                    raise ValueError("sliced or encoded FP8 shards require explicit decoding before reconstruction")
                if event["state"] != "tensor" or entry["manifest"].get("status") != "complete" or entry["errors"]:
                    raise ValueError("shard payload or capture is incomplete")
                span = event.get("parameter_flat_range")
                if not isinstance(span, list) or len(span) != 2:
                    raise ValueError("parameter_flat_range is required")
                array = _load_tensor(entry["dir"], event)
                segments.append((*span, array))
                record["sources"].append({"rank": rank, "event_id": event.get("event_id"), "flat_range": span,
                                          "coordinates": coords})
            tensor = ContiguousTensor(shape, segments)
            needed = tensor.size * tensor.dtype.itemsize
            if total_bytes + needed > max_bytes:
                raise ValueError("reconstruction exceeds explicit max_bytes budget")
            total_bytes += needed
            record.update(status="complete", original_shape=list(shape), dtype=dtype, bytes=needed)
            prepared.append((record, first, tensor))
        except (ValueError, TypeError, OSError) as exc:
            record["reason"] = str(exc)
        records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    # Each TP/PP coordinate retains a distinct synthetic rank; merging its DP
    # shards must never collapse different tensor-parallel parameters together.
    topologies = sorted({json.dumps(record["coordinates"], sort_keys=True) for record, _, _ in prepared})
    topology_ranks = {key: rank for rank, key in enumerate(topologies)}
    rank_events = defaultdict(list)
    for record, first, tensor in prepared:
        rank = topology_ranks[json.dumps(record["coordinates"], sort_keys=True)]
        directory = output / f"rank_{rank:05d}"
        directory.mkdir(exist_ok=True)
        events = rank_events[rank]
        index = len(events)
        filename = f"tensor_{index:08d}.npy"
        tensor.save(directory / filename)
        event = {k: v for k, v in first.items() if not k.startswith("_")}
        for key in ("parameter_flat_range", "gradient_parallel_coordinates", "file", "stats"):
            event.pop(key, None)
        event.update(event_id=index, file=filename, shape=list(tensor.shape), gradient_scope="full_parameter",
                     stats={"numel": tensor.size}, reconstruction=record["sources"],
                     reconstruction_coordinates=record["coordinates"],
                     reconstruction_scope="complete_flat_parameter_within_fixed_tp_pp_coordinates")
        events.append(event)
    complete = bool(records) and not source["errors"] and all(r["status"] == "complete" for r in records)
    status = "complete" if complete else "incomplete"
    if not rank_events:
        rank_events[0] = []
    for rank, events in rank_events.items():
        directory = output / f"rank_{rank:05d}"
        directory.mkdir(exist_ok=True)
        (directory / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in events), encoding="utf-8")
        coordinates = json.loads(topologies[rank]) if topologies else None
        write_json(directory / "manifest.json", {"schema_version": 2, "rank": rank, "world_size": len(rank_events),
            "status": status, "events_count": len(events), "config": {},
            "metadata": {"reconstructed": True, "source": str(Path(capture).resolve()), "scope": "fixed_tp_pp_flat_parameter",
                         "parallelism": {"reconstruction_coordinates": coordinates}}})
    result = {"analysis_schema_version": 1, "kind": "shard_reconstruction", "status": status,
              "records": records, "errors": source["errors"], "output": str(output.resolve()), "bytes": total_bytes,
              "scope": "Exact continuous flat ranges only. TP/PP groups remain separate; no axis inference or missing-shard fill."}
    write_json(output / "reconstruction.json", result)
    return result
