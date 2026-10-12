"""Offline configuration comparison, mapping suggestions and observed trends."""
from __future__ import annotations

from collections import Counter, defaultdict
import difflib
import json
import math
from pathlib import Path

from .compare import (_KEY_FIELDS, _GRADIENT_LAYOUT_FIELDS, _capture_summary, _dtype_name,
                      _identity, _nonfinite_counts, _read_capture)


def protect_output(output, inputs):
    if output is None:
        return
    target = Path(output).expanduser().resolve()
    for source in inputs:
        if isinstance(source, (str, Path)):
            source = Path(source).expanduser().resolve()
            if target == source or (source.is_dir() and target.is_relative_to(source)):
                raise ValueError("analysis outputs must be outside all input artifacts")


def write_json(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _configs(source):
    if isinstance(source, dict):
        return {"document": source}, []
    path = Path(source).expanduser()
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("configuration must contain an object")
        return {"document": value}, []
    capture = _read_capture(path)
    return {str(rank): entry["manifest"] for rank, entry in capture["ranks"].items()}, capture["errors"]


def compare_configs(reference, candidate, *, fields=None, output=None):
    """equal means equality of requested fields, never numerical correctness."""
    protect_output(output, (reference, candidate))
    fields = fields if fields is not None else [
        "metadata.python_version", "metadata.torch_version", "metadata.torch_musa_version",
        "metadata.driver_version", "metadata.parallelism", "metadata.training",
        "metadata.transformer_engine_version", "metadata.precision", "metadata.input_state", "metadata.random_state"]
    if (not isinstance(fields, (tuple, list)) or not fields
            or any(not isinstance(field, str) or not field or any(not part for part in field.split(".")) for field in fields)):
        raise ValueError("fields must be a nonempty list of dotted paths")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate configuration field")
    left, le = _configs(reference)
    right, re = _configs(candidate)
    absent = object()
    def lookup(doc, field):
        for part in field.split("."):
            if not isinstance(doc, dict) or part not in doc:
                return absent
            doc = doc[part]
        return doc
    def compare_value(a, b, path):
        def unavailable(value):
            return (value is absent or value is None
                    or isinstance(value, dict) and (not value or value.get("status") == "unknown"))
        if unavailable(a) or unavailable(b):
            return [], [path]
        if type(a) is not type(b):
            return [path], []
        if isinstance(a, dict):
            pairs = [(a.get(key, absent), b.get(key, absent), path + "." + str(key))
                     for key in sorted(set(a) | set(b), key=str)]
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return [path], []
            pairs = [(av, bv, f"{path}[{index}]") for index, (av, bv) in enumerate(zip(a, b))]
        else:
            return ([] if a == b else [path]), []
        different, unknown = [], []
        for av, bv, child in pairs:
            child_different, child_unknown = compare_value(av, bv, child)
            different.extend(child_different)
            unknown.extend(child_unknown)
        return different, unknown
    records = []
    for rank in sorted(set(left) | set(right)):
        for field in fields:
            a, b = lookup(left.get(rank), field), lookup(right.get(rank), field)
            different, unknown = compare_value(a, b, field)
            status = "different" if different else ("unknown" if unknown else "equal")
            records.append({"rank": rank, "field": field, "status": status,
                            "reference": None if a is absent else a, "candidate": None if b is absent else b,
                            "reference_present": a is not absent, "candidate_present": b is not absent,
                            "different_paths": different, "unknown_paths": unknown})
    counts = Counter(record["status"] for record in records)
    overall = "different" if counts["different"] else ("unknown" if counts["unknown"] or le or re or not records else "equal")
    result = {"analysis_schema_version": 1, "kind": "configuration_comparison", "overall": overall,
              "fields": list(fields), "summary": dict(counts), "records": records,
              "errors": {"reference": le, "candidate": re},
              "scope": "Only requested configuration fields; equality does not establish model precision."}
    if output is not None:
        write_json(output, result)
    return result


def suggest_mapping(reference, candidate, *, output=None, max_candidates=5, max_pairs=100000):
    """Return review-only candidates. Never accept or numerically compare a pair."""
    protect_output(output, (reference, candidate))
    if type(max_candidates) is not int or not 1 <= max_candidates <= 100 or type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("invalid mapping suggestion budget")
    left, right = _read_capture(reference), _read_capture(candidate)
    context = ("step", "microbatch", "phase", "kind", "tensor_path", "recompute", "config_version", "backward_id", "schema_version", "virtual_pipeline", "op", "optimizer_type", "state_stage")
    def signature(event):
        data = {key: event.get(key, 1 if key == "schema_version" else (False if key == "recompute" else None)) for key in context}
        data.update(shape=event["shape"], dtype=_dtype_name(event["dtype"]), tensor_slice=event.get("tensor_slice"),
                    gradient_layout={k: event[k] for k in _GRADIENT_LAYOUT_FIELDS if k in event})
        return json.dumps(data, sort_keys=True, allow_nan=False)
    suggestions, examined, exhausted, rejected_ranks = [], 0, False, []
    for rank in sorted(left["ranks"]):
        manifests = [data["ranks"].get(rank, {}).get("manifest", {}) for data in (left, right)]
        layouts = [{"world_size": m.get("world_size"), "parallelism": (m.get("metadata") or {}).get("parallelism")} for m in manifests]
        if layouts[0] != layouts[1]:
            rejected_ranks.append({"rank": rank, "reason": "parallel layout differs or rank is missing", "layouts": layouts})
            continue
        refs = left["ranks"][rank]["events"]
        cands = right["ranks"].get(rank, {}).get("events", [])
        ref_counts, cand_counts = Counter(_identity(e) for e in refs), Counter(_identity(e) for e in cands)
        groups = defaultdict(list)
        for event in cands:
            if not (ref_counts[_identity(event)] == cand_counts[_identity(event)] == 1):
                groups[signature(event)].append(event)
        for event in refs:
            if ref_counts[_identity(event)] == cand_counts[_identity(event)] == 1:
                continue
            scored = []
            for other in groups[signature(event)]:
                if examined >= max_pairs:
                    exhausted = True
                    break
                examined += 1
                similarity = difflib.SequenceMatcher(None, event["module"], other["module"], autojunk=False).ratio()
                index_delta = abs(event["call_index"] - other["call_index"])
                score = .6 + .3 * similarity + .1 / (1 + index_delta)
                scored.append({"event_id": other.get("event_id"), "module": other["module"],
                               "call_index": other["call_index"], "score": round(score, 6),
                               "evidence": {"context_shape_dtype_op_layout_equal": True,
                                            "module_name_similarity": similarity, "call_index_distance": index_delta}})
            scored.sort(key=lambda x: (-x["score"], x["event_id"] if x["event_id"] is not None else -1))
            suggestions.append({"rank": rank, "reference_event_id": event.get("event_id"),
                "reference_identity": dict(zip(_KEY_FIELDS, _identity(event))), "accepted": False,
                "status": "budget_exhausted" if exhausted else ("ambiguous" if len(scored) > 1 else ("candidate" if scored else "unmatched")),
                "candidate_count": len(scored), "candidates": scored[:max_candidates]})
    result = {"analysis_schema_version": 1, "kind": "mapping_suggestions", "accepted": False,
              "requires_explicit_mapping": True, "pair_budget": max_pairs, "examined_pairs": examined,
              "truncated": exhausted, "suggestions": suggestions, "rejected_ranks": rejected_ranks,
              "coverage": {"reference": _capture_summary(left), "candidate": _capture_summary(right)},
              "scope": "Metadata candidates only. Even a unique candidate is not automatically accepted or precision-verified."}
    if output is not None:
        write_json(output, result)
    return result


def build_trends(capture, *, output=None):
    """Plot recorded statistics, preserving missing steps and duplicate identities."""
    protect_output(output, (capture,))
    data = _read_capture(capture)
    series = []
    metric_names = ("min", "max", "mean", "l2", "nan_count", "posinf_count", "neginf_count")
    fields = tuple(k for k in _KEY_FIELDS if k != "step")
    for rank, entry in sorted(data["ranks"].items()):
        groups = defaultdict(list)
        for event in entry["events"]:
            key = {k: event.get(k, 1 if k == "schema_version" else (False if k == "recompute" else None)) for k in fields}
            key.update(shape=event["shape"], dtype=event["dtype"], tensor_slice=event.get("tensor_slice"),
                       gradient_layout={k: event[k] for k in _GRADIENT_LAYOUT_FIELDS if k in event})
            groups[json.dumps(key, sort_keys=True)].append(event)
        observed = {e["step"] for e in entry["events"]}
        selected = (entry["manifest"].get("config") or {}).get("steps")
        if isinstance(selected, list) and selected and all(type(s) is int and s >= 0 for s in selected):
            expected = sorted(set(selected) | observed)
        elif observed:
            if max(observed) - min(observed) > 100000:
                raise ValueError("trend step span exceeds 100000; use explicit sampled steps")
            expected = list(range(min(observed), max(observed) + 1))
        else:
            expected = []
        for encoded, events in groups.items():
            by_step = defaultdict(list)
            for event in events:
                by_step[event["step"]].append(event)
            points = []
            for step in expected:
                choices = by_step[step]
                point = {"step": step, "state": "missing", "metrics": {}, "event_ids": [e.get("event_id") for e in choices]}
                if len(choices) > 1:
                    point["state"] = "duplicate"
                elif choices:
                    event = choices[0]
                    if event["state"] in ("tensor", "statistics"):
                        metrics = {k: event.get("stats", {}).get(k) for k in metric_names}
                        point.update(metrics=metrics, state="nonfinite" if any(_nonfinite_counts(event).values()) else "available")
                        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in metrics.values()):
                            point.update(state="statistics_invalid", metrics={k: v if type(v) in (int, float) and math.isfinite(v) else None for k, v in metrics.items()})
                        if not any(value is not None for value in metrics.values()):
                            point["state"] = "statistics_missing"
                    else:
                        point["state"] = event["state"]
                points.append(point)
            series.append({"rank": rank, "identity": json.loads(encoded), "points": points})
    counts = Counter(p["state"] for s in series for p in s["points"])
    complete = bool(series) and not data["errors"] and all(e["manifest"].get("status") == "complete" for e in data["ranks"].values())
    complete = complete and not any(counts[s] for s in set(counts) - {"available", "nonfinite"})
    result = {"analysis_schema_version": 1, "kind": "trends", "status": "complete" if complete else "incomplete",
              "series": series, "summary": dict(counts), "coverage": _capture_summary(data),
              "scope": "Recorded statistics only. Missing/truncated/duplicate points are gaps, never zero or interpolated success."}
    if output is not None:
        output = Path(output)
        write_json(output / "trends.json", result)
        from .report import render_trends
        render_trends(result, output / "trends.html")
    return result
