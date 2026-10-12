"""Observed Tensor-version dependencies, never module-name-derived edges."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .analysis import protect_output, write_json
from .compare import _capture_summary, _nonfinite_counts, _public_event, _read_capture


def analyze_graph(capture, *, output=None):
    protect_output(output, (capture,))
    data = _read_capture(capture)
    nodes, edges, unknown, errors, unrepresented = [], [], [], list(data["errors"]), []
    for rank, entry in sorted(data["ranks"].items()):
        observed = [e for e in entry["events"] if e.get("graph_capture") is True]
        non_tensor_ids = set()
        for event in observed:
            if (event.get("tensor_id") is None and event["state"] == "unsupported" and event.get("dtype") is None
                    and str(event.get("reason", "")).startswith("unsupported value type:")):
                non_tensor_ids.add(id(event))
                unrepresented.append({"rank": rank, "event_id": event.get("event_id"), "kind": event["kind"],
                                      "op": event.get("op"), "tensor_path": event["tensor_path"], "reason": event["reason"]})
        observed = [event for event in observed if id(event) not in non_tensor_ids]
        producers, event_ids = defaultdict(list), defaultdict(list)
        for event in observed:
            if event["kind"] == "api_output":
                token = event.get("tensor_id")
                if not isinstance(token, str) or not token or type(event.get("event_id")) is not int:
                    errors.append({"rank": rank, "reason": "graph output lacks valid tensor_id/event_id"})
                    continue
                producers[token].append(event)
                event_ids[event["event_id"]].append(event)
        valid = {}
        for token, choices in producers.items():
            event = choices[0]
            if len(choices) != 1 or len(event_ids[event["event_id"]]) != 1:
                errors.append({"rank": rank, "tensor_id": token, "reason": "duplicate graph tensor/event identity"})
                continue
            valid[token] = event
            nodes.append({"id": f"r{rank}:e{event['event_id']}", "rank": rank,
                          "tensor_id": token, "event_id": event["event_id"],
                          "nonfinite": _nonfinite_counts(event), "event": _public_event(event)})
        inputs, invalid_claims = defaultdict(list), set()
        for event in observed:
            if event["kind"] != "api_input":
                continue
            token, producer_id = event.get("tensor_id"), event.get("producer_event_id")
            if not isinstance(token, str) or not token:
                errors.append({"rank": rank, "event_id": event.get("event_id"), "reason": "graph input lacks tensor_id"})
                continue
            inputs[token].append(event)
            if producer_id is not None:
                source = valid.get(token)
                if (source is None or source.get("event_id") != producer_id
                        or source["_line_number"] >= event["_line_number"]):
                    errors.append({"rank": rank, "event_id": event.get("event_id"), "reason": "input producer claim is not supported by an earlier observed output"})
                    invalid_claims.add(id(event))
        for token, event in valid.items():
            target = f"r{rank}:e{event['event_id']}"
            dependencies = event.get("input_tensor_ids")
            if not isinstance(dependencies, list) or any(not isinstance(t, str) or not t for t in dependencies):
                errors.append({"rank": rank, "event_id": event["event_id"], "reason": "graph output lacks valid input_tensor_ids"})
                continue
            for dependency in sorted(set(dependencies)):
                producer = valid.get(dependency)
                # An input record describing this exact invocation can invalidate
                # a token after unobserved mutation even if an older token exists.
                call_inputs = [e for e in inputs[dependency] if all(e.get(k) == event.get(k) for k in
                               ("api_call_index", "step", "microbatch", "phase", "config_version", "backward_id", "virtual_pipeline"))
                               and e["_line_number"] < event["_line_number"]]
                if any(id(e) in invalid_claims for e in call_inputs):
                    unknown.append({"rank": rank, "target": target, "tensor_id": dependency, "reason": "invalid_producer_claim"})
                    continue
                explicit_unknown = [e for e in call_inputs if e.get("graph_unknown_reason")]
                if explicit_unknown:
                    unknown.append({"rank": rank, "target": target, "tensor_id": dependency,
                                    "reason": explicit_unknown[-1]["graph_unknown_reason"]})
                    continue
                if producer is None:
                    unknown.append({"rank": rank, "target": target, "tensor_id": dependency, "reason": "producer_not_observed"})
                elif producer["_line_number"] >= event["_line_number"]:
                    errors.append({"rank": rank, "target": target, "tensor_id": dependency, "reason": "producer occurs after consumer or creates a self-edge"})
                else:
                    edges.append({"source": f"r{rank}:e{producer['event_id']}", "target": target,
                                  "tensor_id": dependency, "evidence": "observed_tensor_version"})
    collectives, collective_errors = defaultdict(list), []
    for rank, entry in sorted(data["ranks"].items()):
        for event in entry["events"]:
            if event["kind"] not in ("communication_input", "communication_output"):
                continue
            communication = event.get("communication")
            if not isinstance(communication, dict):
                collective_errors.append({"rank": rank, "reason": "communication event lacks explicit metadata"})
                continue
            group, sequence = communication.get("group_id"), communication.get("sequence")
            if not isinstance(group, str) or not group or type(sequence) is not int or sequence < 0:
                collective_errors.append({"rank": rank, "reason": "invalid communication group_id/sequence"})
                continue
            collectives[(group, sequence)].append((rank, event, communication))
    for (group, sequence), members in collectives.items():
        base = members[0][2]
        participants = base.get("participants")
        issue = None
        if (not isinstance(participants, list) or not participants or any(type(r) is not int or r < 0 for r in participants)
                or len(set(participants)) != len(participants) or base.get("operation") != "all_reduce"
                or base.get("synchronous") is not True or not isinstance(base.get("reduction"), str)):
            issue = "invalid explicit collective metadata"
        else:
            expected = {(r, stage) for r in participants for stage in ("input", "output")}
            seen = Counter((rank, c.get("stage")) for rank, _, c in members)
            common = ("operation", "group_id", "sequence", "reduction", "synchronous")
            if set(seen) != expected or any(v != 1 for v in seen.values()):
                issue = "collective requires exactly one input and output from every explicit participant"
            elif any(any(c.get(k) != base.get(k) for k in common) or c.get("participants") != participants
                     or e["kind"] != "communication_" + str(c.get("stage")) for _, e, c in members):
                issue = "collective participant metadata conflicts"
            elif any((e.get("step"), e.get("microbatch"), e.get("config_version"), e.get("backward_id")) !=
                     tuple(members[0][1].get(k) for k in ("step", "microbatch", "config_version", "backward_id")) for _, e, _ in members):
                issue = "collective execution context conflicts"
            elif any(e["shape"] != members[0][1]["shape"] or e["dtype"] != members[0][1]["dtype"] for _, e, _ in members):
                issue = "collective tensor shape/dtype conflicts"
            else:
                by_rank_stage = {(r, c["stage"]): e for r, e, c in members}
                if any(by_rank_stage[(r, "input")]["_line_number"] >= by_rank_stage[(r, "output")]["_line_number"] for r in participants):
                    issue = "collective output must follow its input on every rank"
        if issue:
            collective_errors.append({"group_id": group, "sequence": sequence, "reason": issue,
                                      "observed": [{"rank": r, "stage": c.get("stage"), "event_id": e.get("event_id")} for r, e, c in members]})
            continue
        collective_nodes = {"input": [], "output": []}
        for rank, event, communication in members:
            if type(event.get("event_id")) is not int:
                collective_errors.append({"group_id": group, "sequence": sequence, "reason": "communication event lacks event_id"})
                break
            node_id = f"r{rank}:c:{group}:{sequence}:{communication['stage']}"
            nodes.append({"id": node_id, "rank": rank, "tensor_id": event.get("tensor_id"), "event_id": event["event_id"],
                          "nonfinite": _nonfinite_counts(event), "event": _public_event(event)})
            collective_nodes[communication["stage"]].append(node_id)
        else:
            for source in collective_nodes["input"]:
                for target in collective_nodes["output"]:
                    edges.append({"source": source, "target": target, "evidence": "observed_synchronous_collective",
                                  "group_id": group, "sequence": sequence, "participants": participants, "reduction": base["reduction"]})
    errors.extend(collective_errors)
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge["target"]].append(edge["source"])
    anomalous = {node["id"] for node in nodes if any(node["nonfinite"].values())}
    # These are boundaries in observed edges, not causal or full communication roots.
    boundaries = sorted(node for node in anomalous if not any(parent in anomalous for parent in incoming[node]))
    unknown_targets = {item["target"] for item in unknown}
    incomplete_events = [{"rank": rank, "event_id": event.get("event_id"), "state": event["state"]} for rank, entry in data["ranks"].items()
                         for event in entry["events"] if event["state"] in ("truncated", "error")]
    complete = bool(nodes) and not errors and not unknown and not unrepresented and not incomplete_events and all(e["manifest"].get("status") == "complete" for e in data["ranks"].values())
    result = {"analysis_schema_version": 1, "kind": "observed_dependency_graph", "status": "complete" if complete else ("incomplete" if collective_errors else "partial"),
              "nodes": nodes, "edges": edges, "unknown_inputs": unknown, "errors": errors, "unrepresented_values": unrepresented,
              "incomplete_events": incomplete_events,
              "observed_nonfinite_boundaries": [{"node": node, "has_unknown_input": node in unknown_targets} for node in boundaries],
              "summary": {"nodes": len(nodes), "edges": len(edges), "unknown_inputs": len(unknown), "errors": len(errors),
                          "nonfinite_nodes": len(anomalous), "unrepresented_values": len(unrepresented)}, "coverage": _capture_summary(data),
              "scope": "Observed ATen Tensor-version edges and explicitly recorded complete synchronous all_reduce groups only. Unobserved producers/aliases remain unknown. Nonfinite boundaries are observations, not root-cause proof."}
    if output is not None:
        output = Path(output)
        write_json(output / "graph.json", result)
        from .report import render_graph
        render_graph(result, output / "graph.html")
    return result
