"""Declarative comparison policies. No expressions or executable plugins are loaded."""
from __future__ import annotations

import fnmatch
import json
import math
from pathlib import Path


def load_object(value, *, name="configuration"):
    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object or JSON file")
    return value


def validate_tolerance(name, value):
    if value is None:
        return
    if name == "max_ulp":
        if type(value) is not int or value < 0:
            raise ValueError("max_ulp must be a nonnegative integer")
    elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


class ThresholdRules:
    def __init__(self, value=None):
        self.rules = []
        if value is None:
            return
        value = load_object(value, name="threshold rules")
        if set(value) - {"schema_version", "rules"} or value.get("schema_version", 1) != 1:
            raise ValueError("invalid threshold rules schema")
        rules = value.get("rules")
        if not isinstance(rules, list):
            raise ValueError("threshold rules must contain a rules list")
        ids = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or set(rule) - {"id", "module", "op", "dtype", "atol", "rtol", "max_ulp"}:
                raise ValueError("unknown threshold rule field")
            item = dict(rule)
            item.setdefault("id", f"rule_{index}")
            if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
                raise ValueError("threshold rule ids must be unique nonempty strings")
            ids.add(item["id"])
            if not any(k in item for k in ("module", "op", "dtype")):
                raise ValueError("threshold rule requires a module, op or dtype selector")
            for key in ("module", "op", "dtype"):
                if key in item and (not isinstance(item[key], str) or not item[key]):
                    raise ValueError("threshold selectors must be nonempty glob strings")
            if not any(k in item for k in ("atol", "rtol", "max_ulp")):
                raise ValueError("threshold rule has no threshold")
            for key in ("atol", "rtol", "max_ulp"):
                if key in item:
                    if item[key] is None:
                        raise ValueError("rule thresholds cannot be null")
                    validate_tolerance(key, item[key])
            self.rules.append(item)

    def resolve(self, event, *, atol=None, rtol=None, max_ulp=None):
        resolved, owners, matched = {}, {}, []
        for rule in self.rules:
            if not all(fnmatch.fnmatchcase(str(event.get(k) or ""), rule[k])
                       for k in ("module", "op", "dtype") if k in rule):
                continue
            matched.append(rule["id"])
            for key in ("atol", "rtol", "max_ulp"):
                if key not in rule:
                    continue
                if key in resolved and resolved[key] != rule[key]:
                    raise ValueError(f"conflicting {key} rules: {owners[key]} and {rule['id']}")
                resolved[key], owners[key] = rule[key], rule["id"]
        for key, value in (("atol", atol), ("rtol", rtol), ("max_ulp", max_ulp)):
            validate_tolerance(key, value)
            if value is not None:
                resolved[key], owners[key] = value, "explicit"
        return resolved, {"matched_rules": matched, "sources": owners}


class ExplicitMapping:
    """Candidate names become reference names; context fields cannot be rewritten."""
    REMAPPABLE = {"module", "op", "parameter_name", "tensor_path", "call_index", "api_call_index"}
    CONTEXT = {"rank", "step", "microbatch", "phase", "kind", "recompute", "config_version", "backward_id", "virtual_pipeline"}

    def __init__(self, value=None):
        self.pairs = []
        self.uses = {"reference": {}, "candidate": {}}
        if value is None:
            return
        value = load_object(value, name="explicit mapping")
        if set(value) - {"schema_version", "pairs"} or value.get("schema_version", 1) != 1:
            raise ValueError("invalid mapping schema")
        if not isinstance(value.get("pairs"), list):
            raise ValueError("mapping requires a pairs list")
        ids, seen = set(), {"reference": set(), "candidate": set()}
        for index, pair in enumerate(value["pairs"]):
            if not isinstance(pair, dict) or set(pair) - {"id", "reference", "candidate"}:
                raise ValueError("unknown mapping field")
            pair = dict(pair)
            pair.setdefault("id", f"mapping_{index}")
            if not isinstance(pair["id"], str) or not pair["id"] or pair["id"] in ids:
                raise ValueError("mapping ids must be unique nonempty strings")
            ids.add(pair["id"])
            for side in ("reference", "candidate"):
                selector = pair.get(side)
                if not isinstance(selector, dict) or not selector or set(selector) - (self.REMAPPABLE | self.CONTEXT):
                    raise ValueError("mapping selectors require exact identity fields")
                if any(not isinstance(v, (str, int, bool, type(None))) for v in selector.values()):
                    raise ValueError("mapping selector values must be scalars")
                encoded = json.dumps(selector, sort_keys=True)
                if encoded in seen[side]:
                    raise ValueError(f"mapping is not one-to-one: repeated {side} selector")
                seen[side].add(encoded)
            left, right = pair["reference"], pair["candidate"]
            if set(left) != set(right):
                raise ValueError("mapping sides must specify the same fields")
            if any(left[k] != right[k] for k in self.CONTEXT & left.keys()):
                raise ValueError("mapping cannot change rank, phase, step or other execution context")
            self.pairs.append(pair)

    def apply(self, event, rank, side):
        source = {**event, "rank": rank}
        matches = [p for p in self.pairs if all(k in source and source[k] == v for k, v in p[side].items())]
        if len(matches) > 1:
            raise ValueError("overlapping mapping selectors match one event")
        if not matches:
            return event
        pair = matches[0]
        self.uses[side][pair["id"]] = self.uses[side].get(pair["id"], 0) + 1
        event = dict(event)
        event["_mapping_id"] = pair["id"]
        if side == "candidate":
            event["_mapped_fields"] = {k: v for k, v in pair["reference"].items() if k in self.REMAPPABLE}
        return event
