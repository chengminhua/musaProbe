"""Safe NumPy checkpoint bundles and an explicit, lazy PyTorch converter."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

from .analysis import protect_output, write_json
from .compare import _dtype_name, _load_tensor, _metrics, _tolerances, _write_outputs
from .layout import ContiguousTensor, validate_shape
from .policies import load_object, validate_tolerance


def _read_bundle(source):
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / "checkpoint.json"
    if path.suffix != ".json":
        raise ValueError("checkpoint comparison requires a converted checkpoint.json bundle")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("format") != "musaprobe_checkpoint" or data.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint bundle format")
    if not isinstance(data.get("parameters"), list):
        raise ValueError("checkpoint parameters must be a list")
    groups, errors = defaultdict(list), []
    for index, item in enumerate(data["parameters"]):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            errors.append({"index": index, "reason": "parameter requires a nonempty name"})
            continue
        groups[item["name"]].append(item)
    return path.parent, data, groups, errors


def _parameter(directory, items):
    for event in items:
        validate_shape(event.get("shape"))
        if not isinstance(event.get("dtype"), str):
            raise ValueError("checkpoint parameter dtype must be a string")
    layouts = [item.get("layout") for item in items]
    if not any(layout is not None for layout in layouts):
        if len(items) != 1:
            raise ValueError("duplicate unsharded parameter name")
        event = items[0]
        if event.get("fp8") is not None:
            raise TypeError("encoded FP8 checkpoint conversion is outside the checkpoint bundle contract")
        if event.get("state", "tensor") != "tensor" or event.get("sharded", False):
            raise ValueError("parameter is unsupported or lacks explicit shard layout")
        array = _load_tensor(directory, event)
        return array, event["dtype"], {"layout": "unsharded"}
    if any(not isinstance(layout, dict) for layout in layouts):
        raise ValueError("mixed full and sharded parameter entries")
    shape, logical_dtype, coordinates, segments = None, None, set(), []
    for event, layout in zip(items, layouts):
        if layout.get("format") != "megatron_contiguous_flat_v1":
            raise ValueError("unsupported shard layout; exact contiguous-flat metadata is required")
        current_shape = validate_shape(layout.get("original_shape"))
        current_dtype = _dtype_name(event.get("dtype"))
        coords = layout.get("parallel_coordinates")
        if not isinstance(coords, dict) or not {"tensor", "pipeline", "data"}.issubset(coords):
            raise ValueError("shard layout requires tensor/pipeline/data coordinates")
        if any(type(v) is not int or v < 0 for v in coords.values()):
            raise ValueError("invalid parallel coordinates")
        encoded = json.dumps(coords, sort_keys=True)
        if encoded in coordinates:
            raise ValueError("duplicate shard parallel coordinates")
        coordinates.add(encoded)
        if shape is not None and (shape != current_shape or logical_dtype != current_dtype):
            raise ValueError("shards disagree on global shape or dtype")
        shape, logical_dtype = current_shape, current_dtype
        span = layout.get("flat_range")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("shard flat_range is required")
        if event.get("state", "tensor") != "tensor" or event.get("tensor_slice") is not None or event.get("fp8") is not None:
            raise ValueError("shard payload is incomplete or encoded")
        segments.append((*span, _load_tensor(directory, event)))
    return ContiguousTensor(shape, segments), logical_dtype, {"layout": "megatron_contiguous_flat_v1",
        "shards": [{"file": i.get("file"), **i["layout"]} for i in items]}


def compare_checkpoints(reference, candidate, *, atol=None, rtol=None, max_ulp=None, output=None):
    """Compare named complete tensor values; never restore training state."""
    protect_output(output, tuple(Path(source).parent if Path(source).suffix == ".json" else source for source in (reference, candidate)))
    for key, value in (("atol", atol), ("rtol", rtol), ("max_ulp", max_ulp)):
        validate_tolerance(key, value)
    rd, rm, refs, re = _read_bundle(reference)
    cd, cm, cands, ce = _read_bundle(candidate)
    records = []
    for side, errors in (("reference", re), ("candidate", ce)):
        for error in errors:
            records.append({"rank": None, "identity": {}, "status": "error", "reason": f"{side}: {error}", "metrics": {}})
    for name in sorted(set(refs) | set(cands)):
        record = {"rank": 0, "identity": {"module": name, "parameter_name": name, "phase": "checkpoint",
                  "kind": "parameter", "tensor_path": "$"}, "metrics": {}, "reference": refs.get(name), "candidate": cands.get(name)}
        try:
            if name not in refs or name not in cands:
                record.update(status="missing_reference" if name not in refs else "missing_candidate", reason="Parameter is missing.")
            else:
                a, ad, al = _parameter(rd, refs[name])
                b, bd, bl = _parameter(cd, cands[name])
                record["layouts"] = {"reference": al, "candidate": bl}
                if a.shape != b.shape:
                    record.update(status="shape_mismatch", reason="Complete parameter shapes differ.")
                elif _dtype_name(ad) != _dtype_name(bd):
                    record.update(status="dtype_mismatch", reason="Logical parameter dtypes differ.")
                else:
                    aa, rr = _tolerances(ad, atol, rtol)
                    metrics = _metrics(a, b, aa, rr, compute_ulp=_dtype_name(ad) == "float32", max_ulp=max_ulp)
                    record["metrics"] = metrics
                    nonfinite = metrics["nonfinite_reference"] or metrics["nonfinite_candidate"]
                    record.update(status="nonfinite" if nonfinite else ("mismatch" if metrics["mismatch_count"] else "pass"),
                                  reason="Complete named parameter compared with enabled rules.")
        except (ValueError, TypeError, OSError, KeyError) as exc:
            record.update(status="error", reason=str(exc))
        records.append(record)
    counts = Counter(r["status"] for r in records)
    failed = any(counts[s] for s in ("error", "mismatch", "nonfinite", "shape_mismatch", "dtype_mismatch"))
    incomplete = not records or bool(counts["missing_reference"] or counts["missing_candidate"])
    # Conversion failures remain evidence even if all readable names compare.
    source_errors = {"reference": rm.get("unsupported", []), "candidate": cm.get("unsupported", [])}
    incomplete = incomplete or bool(source_errors["reference"] or source_errors["candidate"])
    incomplete = incomplete or any(m.get("status", "complete") != "complete" for m in (rm, cm))
    result = {"schema_version": 1, "kind": "checkpoint_comparison", "overall": "fail" if failed else ("incomplete" if incomplete else "pass"),
              "records": records, "summary": {"statuses": dict(counts), "records": len(records)},
              "reference": str(Path(reference).resolve()), "candidate": str(Path(candidate).resolve()),
              "coverage": {"unsupported_conversion_entries": source_errors},
              "first_difference": next((r for r in records if r["status"] in {"mismatch", "nonfinite", "shape_mismatch", "dtype_mismatch"}), None),
              "notes": ["Named converted tensor values only; arbitrary Megatron axis layouts are not inferred and optimizer/training state is not restored."]}
    if output is not None:
        _write_outputs(result, output)
    return result


def convert_checkpoint(source, output, *, key=None, layout=None, max_bytes=1073741824):
    """Explicit weights-only PyTorch conversion. Torch is imported only here.

    A supplied layout describes each local tensor; values are never inferred
    from file names. `name` may specify the global parameter name.
    """
    protect_output(output, (source,))
    if Path(source).is_dir():
        if layout is not None:
            raise ValueError("native distributed conversion cannot combine an external flat layout")
        if (Path(source) / ".metadata").is_file():
            from .distributed_checkpoint import convert_torch_dist
            return convert_torch_dist(source, output, prefix=key, max_bytes=max_bytes)
        raise ValueError("checkpoint directory is not supported torch_dist metadata; Zarr and legacy rank-directory assembly require separate explicit adapters")
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise ValueError("checkpoint conversion requires a new or empty output directory")
    if type(max_bytes) is not int or max_bytes < 1 or Path(source).stat().st_size > max_bytes:
        raise ValueError("checkpoint exceeds explicit positive max_bytes conversion budget")
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            if sum(info.file_size for info in archive.infolist()) > max_bytes:
                raise ValueError("checkpoint uncompressed bytes exceed max_bytes budget")
    metadata = load_object(layout, name="checkpoint layout") if layout is not None else {"parameters": {}}
    if set(metadata) - {"parameters"} or not isinstance(metadata.get("parameters"), dict):
        raise ValueError("checkpoint layout requires a parameters object")
    try:
        import torch
    except ImportError as exc:
        raise ValueError("checkpoint conversion requires PyTorch; NumPy bundle comparison does not") from exc
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError(f"checkpoint cannot be read with weights_only=True: {exc}") from exc
    if key:
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"checkpoint key does not exist: {key}")
            value = value[part]
    elif isinstance(value, dict):
        wrappers = [k for k in ("state_dict", "model") if isinstance(value.get(k), dict)]
        if len(wrappers) > 1:
            raise ValueError("multiple model dictionaries; specify key explicitly")
        if wrappers:
            value = value[wrappers[0]]
    if not isinstance(value, dict):
        raise ValueError("selected checkpoint entry must be a state dictionary")
    if set(metadata["parameters"]) - set(value):
        raise ValueError("layout contains parameter names absent from selected checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    parameters, unsupported, saved_bytes = [], [], 0
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            unsupported.append({"name": str(name), "reason": "not a named tensor"})
            continue
        try:
            logical = str(tensor.dtype)
            stored_itemsize = 4 if tensor.dtype == torch.bfloat16 else tensor.element_size()
            if saved_bytes + tensor.numel() * stored_itemsize > max_bytes:
                raise ValueError("converted tensor bytes exceed max_bytes budget")
            tensor = tensor.detach().cpu().contiguous()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            array = tensor.numpy()
            if array.dtype.kind not in "biuf":
                raise TypeError("unsupported tensor dtype")
            index = len(parameters)
            item = {"name": name, "file": f"tensor_{index:08d}.npy", "shape": list(array.shape), "dtype": logical}
            if name in metadata["parameters"]:
                spec = metadata["parameters"][name]
                if not isinstance(spec, dict):
                    raise ValueError("layout entry must be an object")
                item["name"] = spec.get("name", name)
                item["layout"] = {k: v for k, v in spec.items() if k != "name"}
            if not isinstance(item["name"], str) or not item["name"]:
                raise ValueError("global parameter name must be nonempty")
            if saved_bytes + array.nbytes > max_bytes:
                raise ValueError("converted tensor bytes exceed max_bytes budget")
            np.save(output / item["file"], array, allow_pickle=False)
            saved_bytes += array.nbytes
            parameters.append(item)
        except (TypeError, ValueError, RuntimeError) as exc:
            unsupported.append({"name": name, "reason": str(exc)})
    result = {"format": "musaprobe_checkpoint", "schema_version": 1, "parameters": parameters,
              "unsupported": unsupported, "source_format": "torch_weights_only", "selected_key": key,
              "status": "complete" if parameters and not unsupported else "incomplete"}
    write_json(output / "checkpoint.json", result)
    return result
