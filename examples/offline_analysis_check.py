#!/usr/bin/env python3
"""CPU-only acceptance over existing GPU captures; originals are never changed."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from musaprobe.analysis import write_json
from musaprobe.fp8_numpy import FP8Tensor


def exercise(capture_root, output, fp8_bundles):
    captures = Path(capture_root).resolve()
    if not (captures / "graph_reference").is_dir():
        choices = list(captures.glob("*/captures"))
        if len(choices) != 1:
            raise ValueError("capture-root must identify exactly one capture_extensions captures directory")
        captures = choices[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "rawlogs"
    logs.mkdir()
    checks, commands, evidence = {}, [], {}

    def check(name, value):
        checks[name] = bool(value)

    def cli(label, *args):
        command = [sys.executable, "-m", "musaprobe", *map(str, args)]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120,
                                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        (logs / f"{label}.stdout.json").write_text(completed.stdout, encoding="utf-8")
        (logs / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        commands.append({"label": label, "argv": command, "exit_code": completed.returncode})
        try:
            data = json.loads(completed.stdout)
        except ValueError:
            data = {}
        return completed.returncode, data

    def fixture(name, source):
        destination = output / "controlled_fixtures" / name
        shutil.copytree(source, destination)
        for manifest in destination.glob("rank_*/manifest.json"):
            data = json.loads(manifest.read_text())
            data.setdefault("metadata", {})["controlled_fixture"] = {"kind": name, "source": str(source)}
            write_json(manifest, data)
        return destination

    reference, repeated, changed = (captures / name for name in ("graph_reference", "graph_repeated", "graph_changed"))
    rc, data = cli("actual_repeat", "compare", reference, repeated, "--output", output / "actual_repeat")
    check("actual_independent_gpu_repeat_pass", rc == 0 and data.get("overall") == "pass")
    rc, data = cli("actual_difference", "compare", reference, changed, "--atol", 0, "--rtol", 0, "--output", output / "actual_difference")
    check("actual_gpu_change_detected", rc == 1 and data.get("first_difference") is not None)
    full = json.loads((output / "actual_difference/comparison.json").read_text())
    tolerance = max((r.get("metrics", {}).get("max_abs_error") or 0 for r in full["records"]), default=0) * 2 + 1
    rules = output / "rules.json"
    write_json(rules, {"rules": [{"id": "controlled_acceptance_tolerance", "module": "*", "atol": tolerance, "rtol": 0}]})
    rc, _ = cli("rules", "compare", reference, changed, "--rules", rules, "--output", output / "rules_report")
    check("explicit_rule_applied_to_actual_values", rc == 0)
    rc, _ = cli("rules_cli_priority", "compare", reference, changed, "--rules", rules, "--atol", 0, "--rtol", 0,
                "--output", output / "rules_cli_priority")
    check("cli_tolerance_has_priority", rc == 1)
    write_json(output / "conflicting_rules.json", {"rules": [{"module": "*", "atol": 1}, {"module": "*", "atol": 2}]})
    rc, _ = cli("rules_conflict", "compare", reference, repeated, "--rules", output / "conflicting_rules.json",
                "--output", output / "rules_conflict")
    check("conflicting_rules_rejected", rc == 2)

    renamed = fixture("metadata_rename_only", reference)
    modules = set()
    for events in renamed.glob("rank_*/events.jsonl"):
        records = [json.loads(line) for line in events.read_text().splitlines() if line]
        for record in records:
            modules.add(record["module"])
            record["module"] = "controlled_rename." + record["module"]
        events.write_text("".join(json.dumps(e) + "\n" for e in records))
    mapping = output / "mapping.json"
    write_json(mapping, {"pairs": [{"reference": {"module": m}, "candidate": {"module": "controlled_rename." + m}} for m in sorted(modules)]})
    rc, _ = cli("unmapped", "compare", reference, renamed, "--output", output / "unmapped")
    check("renamed_fixture_not_silently_aligned", rc == 2)
    rc, _ = cli("mapped", "compare", reference, renamed, "--mapping", mapping, "--output", output / "mapped")
    check("explicit_mapping_compares_original_gpu_payloads", rc == 0)
    rc, suggestions = cli("suggestions", "suggest-mapping", reference, renamed, "--output", output / "suggestions.json")
    check("suggestions_remain_unaccepted", rc == 0 and suggestions.get("accepted") is False)

    config_a, config_b = fixture("config_a", reference), fixture("config_b", reference)
    for path, version in ((config_a, "controlled_version_a"), (config_b, "controlled_version_b")):
        manifest = path / "rank_00000/manifest.json"
        data = json.loads(manifest.read_text())
        data["metadata"]["torch_version"] = version
        write_json(manifest, data)
    rc, data = cli("config_difference", "check-config", config_a, config_b, "--output", output / "configuration.json")
    check("controlled_config_difference_detected", rc == 1 and data.get("overall") == "different")
    write_json(output / "unknown_a.json", {})
    write_json(output / "unknown_b.json", {})
    rc, data = cli("config_unknown", "check-config", output / "unknown_a.json", output / "unknown_b.json")
    check("missing_config_not_equal", rc == 2 and data.get("overall") == "unknown")
    rc, data = cli("actual_same_capture_config", "check-config", reference, reference)
    random_fields = [record for record in data.get("records", []) if record.get("field") == "metadata.random_state"]
    check("actual_same_capture_unrecorded_rng_remains_unknown", rc == 2 and bool(random_fields)
          and all(record.get("status") == "unknown" for record in random_fields))

    rc, trend = cli("actual_trend", "trend", captures / "async", "--output", output / "actual_trend")
    check("actual_gpu_statistics_trend_readable", rc == 0 and bool(trend.get("series")))
    gap = fixture("controlled_step_gaps", captures / "async")
    records = [json.loads(line) for line in (gap / "rank_00000/events.jsonl").read_text().splitlines() if line]
    for i, event in enumerate(records):
        event.update(step=i * 2, call_index=0)
    (gap / "rank_00000/events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in records))
    manifest = json.loads((gap / "rank_00000/manifest.json").read_text())
    manifest.setdefault("config", {})["steps"] = list(range(len(records) * 2))
    write_json(gap / "rank_00000/manifest.json", manifest)
    rc, trend = cli("gap_trend", "trend", gap, "--output", output / "gap_trend")
    check("controlled_missing_steps_remain_gaps", rc == 2 and trend.get("summary", {}).get("missing", 0) > 0)
    rc, graph = cli("actual_graph", "graph", reference, "--output", output / "actual_graph")
    check("actual_gpu_tensor_edges_found", bool(graph.get("edges")) and all(e.get("evidence") == "observed_tensor_version" for e in graph.get("edges", [])))
    check("external_graph_inputs_remain_unknown", rc == 2 and bool(graph.get("unknown_inputs")))

    if fp8_bundles:
        # A CPU-only independent oracle checks actual GPU-produced bit patterns.
        from musaprobe.native_fp8 import load_fp8_bundle, decode_fp8_bundle
        import torch
        torch.set_num_threads(1)
        for index, source in enumerate(fp8_bundles):
            bundle = load_fp8_bundle(source)
            manifest = json.loads((Path(source) / "manifest.json").read_text())
            specs, info = manifest["files"], bundle["quantization"]
            event = {"shape": info["logical_shape"], "dtype": info.get("nominal_dtype", info["format"]), "fp8": info,
                     "file": specs["payload"]["file"], "auxiliary": {k: v["file"] for k, v in specs.items() if k != "payload"}}
            reader = FP8Tensor(Path(source), event)
            actual = reader.flat[:]
            oracle = decode_fp8_bundle(source).numpy().reshape(-1)
            check(f"actual_fp8_{index}_numpy_cpu_oracle_exact", np.array_equal(actual, oracle, equal_nan=True))
            evidence[f"fp8_{index}"] = {"source": str(source), "format": info["format"], "scale_mode": info["scale_mode"],
                                      "shape": info["logical_shape"], "cpu_oracle": "torch float8 decode", "gpu_execution": False}
    result = {"status": "pass" if all(checks.values()) else "fail", "assertions": checks,
              "failed_assertions": [name for name, passed in checks.items() if not passed],
              "assertion_count": len(checks), "commands": commands, "evidence": evidence,
              "source_captures": str(captures), "output": str(output),
              "scope": "CPU analysis of actual GPU artifacts. Renaming/config/step-gap inputs are labelled controlled copied fixtures, not new GPU training runs. No original capture was modified."}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp8-bundle", action="append", default=[])
    args = parser.parse_args()
    result = exercise(args.capture_root, args.output, args.fp8_bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
