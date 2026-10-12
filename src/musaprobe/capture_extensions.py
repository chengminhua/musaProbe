"""Opt-in eager capture controls. These APIs never imply distributed coordination."""
from __future__ import annotations

import contextlib
import copy
import fnmatch
import json
import threading
import weakref
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from .config import ProbeConfig


def function_name(func):
    module = getattr(func, "__module__", None)
    name = getattr(func, "__name__", type(func).__name__)
    # Several public functional APIs (linear, conv2d, ...) are aliases of C
    # builtins whose __module__ points at torch._C rather than their public API.
    if module == "torch.nn.functional" or getattr(torch.nn.functional, name, None) is func:
        return f"torch.nn.functional.{name}"
    if module == "torch":
        return f"torch.{name}"
    owner = getattr(func, "__objclass__", None)
    if owner is not None and issubclass(owner, torch.Tensor):
        return f"torch.Tensor.{name}"
    return f"{module or 'torch.Tensor'}.{name}"


class HighLevelMode(TorchFunctionMode):
    def __init__(self, probe):
        self.probe = probe

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        probe = self.probe
        name = function_name(func)
        if (not probe.active or probe._is_suspended()
                or not any(fnmatch.fnmatchcase(name, pattern) for pattern in probe.config.high_level_apis)):
            return func(*args, **kwargs)
        metadata = probe._api_metadata(name)
        if not probe._selected(metadata):
            return func(*args, **kwargs)
        metadata["api_level"] = "python"
        metadata["stack"] = probe._stack()
        parent = {"name": name, "call_index": metadata["api_call_index"],
                  "module": metadata["module"], "step": metadata["step"],
                  "microbatch": metadata.get("microbatch"), "config_version": metadata["config_version"]}
        if probe.config.capture_inputs:
            probe._capture_structure(args, "args", "python_api_input", metadata)
            probe._capture_structure(kwargs, "kwargs", "python_api_input", metadata)
        parents = probe._python_frames()
        parents.append(parent)
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            probe._notice(metadata, "python_api_output", "output", "error", f"Python API raised {type(exc).__name__}: {exc}")
            raise
        finally:
            parents.pop()
        if probe.config.capture_outputs:
            probe._capture_structure(result, "output", "python_api_output", metadata)
        return result


class CaptureExtensionsMixin:
    def _init_extensions(self):
        self._high_mode = None
        self._pending_config = None
        self._pending_config_error = None
        self._backward_session = None
        self._backward_sequence = 0
        self._backward_id_scope = None
        self._backward_ids = set()
        self._history = deque(maxlen=self.config.history_size)
        self._history_lock = threading.RLock()
        self._trigger_history = deque(maxlen=max(1, self.config.history_size))
        self._stop_requested = False
        self._replacement_rules = []
        self._graph_tensors = {}
        self._graph_sequence = 0

    def _python_frames(self):
        if not hasattr(self._local, "python_frames"):
            self._local.python_frames = []
        return self._local.python_frames

    def _validate_updated_config(self, value):
        candidate = ProbeConfig.from_dict(value)
        for name in ("output_dir", "run_id", "rank", "world_size", "async_write", "async_d2h", "async_queue_size"):
            if getattr(candidate, name) != getattr(self.config, name):
                raise ValueError(f"{name} cannot change during a capture run")
        if self.writer is not None:
            if candidate.max_rank_bytes < self.writer._reserved_bytes:
                raise ValueError("max_rank_bytes cannot be below already reserved snapshots")
            if candidate.max_events < self.writer.events_count:
                raise ValueError("max_events cannot be below already recorded events")
        return candidate

    def request_config_update(self, changes):
        """Queue a validated partial update; return False and retain old config on error."""
        try:
            if not isinstance(changes, dict):
                raise ValueError("configuration update must be an object")
            self._pending_config = self._validate_updated_config({**self.config.to_dict(), **changes})
            self._pending_config_error = None
            return True
        except Exception as exc:
            self._pending_config = None
            self._pending_config_error = f"{type(exc).__name__}: {exc}"
            return False

    def request_config_reload(self, path):
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._pending_config = None
            self._pending_config_error = f"{type(exc).__name__}: {exc}"
            return False
        return self.request_config_update(value)

    def step_boundary(self, step, *, coordinate=None):
        """Caller asserts all computation for the previous step has completed.

        A coordinator is called on every rank, including ranks excluded from data
        capture. It must perform its own timeout/failed-rank handling. Returning
        stop=True requests loop termination; this method does not raise SystemExit.
        """
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self._frames() or self._backward_session is not None:
            raise RuntimeError("step_boundary requires an idle module/backward boundary")
        # D2H statistics/NaN detection finish on the worker. Drain before reading
        # stop_requested so all ranks agree using completed observations.
        if self.writer is not None and self.config.async_d2h and not self.writer.flush():
            self._pending_config_error = "async_d2h snapshot persistence failed before step boundary"
        # A request can be queued before more snapshots consume the budget.
        # Revalidate before consensus so every rank receives a local rejection,
        # rather than applying a configuration that another rank cannot apply.
        if self._pending_config is not None:
            try:
                self._pending_config = self._validate_updated_config(self._pending_config.to_dict())
                if self.writer is not None and not self.writer.flush():
                    raise RuntimeError("snapshot persistence failed before configuration update")
            except Exception as exc:
                self._pending_config = None
                self._pending_config_error = f"{type(exc).__name__}: {exc}"
        proposal = {"step": step, "config_version": self._context["config_version"],
                    "update": self._pending_config.to_dict() if self._pending_config else None,
                    "error": self._pending_config_error, "stop": self.stop_requested}
        if coordinate is not None:
            response = coordinate(copy.deepcopy(proposal))
        elif (self.world_size or 1) > 1 and (proposal["update"] or proposal["stop"] or proposal["error"]):
            response = {**proposal, "accepted": False, "error": "distributed updates/stops require a boundary coordinator"}
        else:
            response = {**proposal, "accepted": proposal["error"] is None}
        if (not isinstance(response, dict) or type(response.get("accepted")) is not bool
                or response.get("step") != step or response.get("config_version") != proposal["config_version"]
                or type(response.get("stop")) is not bool):
            raise ValueError("invalid boundary coordinator response")
        self._pending_config = None
        self._pending_config_error = None
        if response.get("accepted") and not response.get("error") and response.get("update") is not None:
            candidate = self._validate_updated_config(response["update"])
            if self.writer is not None:
                self.writer.flush()
            self._exit_modes()
            self.config = candidate
            self._context["config_version"] += 1
            self._history = deque(self._history, maxlen=candidate.history_size)
            self._trigger_history = deque(self._trigger_history, maxlen=max(1, candidate.history_size))
            if self.writer is not None:
                self.writer.config = candidate
                self.writer.manifest["metadata"].setdefault("config_versions", []).append(
                    {"version": self._context["config_version"], "step": step, "config": candidate.to_dict()})
            if self.active:
                self._enter_modes()
        elif response.get("error"):
            self._notice(dict(self._context, module="<control>", call_index=0, op=None),
                         "config_update", "", "error", str(response["error"]))
        self.set_context(step=step, microbatch=None, phase="unknown", recompute=False)
        # Configuration rejection must not discard an independently coordinated
        # nonfinite stop. Without a coordinator we never stop only one rank.
        accepted_stop = bool(response["stop"] and (coordinate is not None or (self.world_size or 1) == 1))
        if accepted_stop:
            self.stop()
            if self.writer is not None:
                self.writer.flush()
                self.writer.manifest["metadata"]["stopped_at_boundary"] = step
        return {**response, "stop": accepted_stop, "applied_config_version": self._context["config_version"]}

    @property
    def recent_history(self):
        """Bounded event summaries; tensor files remain subject to the normal disk budget."""
        with self._history_lock:
            return copy.deepcopy(list(self._history))

    @property
    def stop_requested(self):
        with self._history_lock:
            return self._stop_requested

    def _observe_event(self, event):
        if event is None or event.get("_observation_deferred"):
            return
        with self._history_lock:
            self._observe_completed_event(event)

    def _observe_completed_event(self, event):
        summary = {key: copy.deepcopy(event.get(key)) for key in
                   ("event_id", "step", "module", "kind", "tensor_path", "shape", "stats", "file", "config_version", "backward_id")}
        stats = event.get("stats", {})
        nonfinite = sum(stats.get(key, 0) for key in ("nan_count", "posinf_count", "neginf_count"))
        if nonfinite and (self.config.history_size or self.config.stop_on_nonfinite):
            trigger = {"event": summary, "history": self.recent_history}
            self._trigger_history.append(trigger)
            if self.writer is not None:
                self.writer.manifest["metadata"]["nonfinite_triggers"] = list(self._trigger_history)
                self.writer.manifest["metadata"]["history_policy"] = "bounded_event_summaries_selected_tensor_scope"
            if self.config.stop_on_nonfinite:
                self._stop_requested = True
        if self.config.history_size:
            self._history.append(summary)

    @contextlib.contextmanager
    def backward_session(self, backward_id=None, *, retain_graph=False):
        """Label an explicit backward invocation. Pass retain_graph to PyTorch separately.

        Hooks can remain attached for another invocation of the same graph. Use
        release_graph() when done; this API does not call backward or alter its flags.
        """
        if self._backward_session is not None:
            raise RuntimeError("nested backward sessions are not supported")
        if backward_id is None:
            backward_id = self._backward_sequence
            self._backward_sequence += 1
        if (type(backward_id) not in (int, str) or (isinstance(backward_id, str) and not backward_id)
                or (type(backward_id) is int and backward_id < 0)):
            raise ValueError("backward_id must be a nonnegative integer or nonempty string")
        if type(retain_graph) is not bool:
            raise ValueError("retain_graph must be bool")
        scope = (self._context["step"], self._context["config_version"])
        if scope != self._backward_id_scope:
            self._backward_id_scope = scope
            self._backward_ids.clear()
        identity = (self._context["microbatch"], backward_id)
        if identity in self._backward_ids:
            raise ValueError("backward_id was already used in this step/microbatch/configuration")
        self._backward_ids.add(identity)
        old = dict(self._context)
        self._backward_session = {"backward_id": backward_id, "retain_graph": retain_graph}
        self._context.update(phase="backward", backward_id=backward_id)
        failed = False
        try:
            yield self
        except BaseException:
            failed = True
            raise
        finally:
            self._context = old
            self._backward_session = None
            # Successful hooks remove themselves as they fire. Clearing all
            # hooks here would also erase other microbatches' pending graphs.
            if failed:
                self.release_graph()
                self.errors.append("backward session failed; pending tensor observation hooks were released")

    def release_graph(self):
        """Remove all pending observation hooks; does not free PyTorch graphs."""
        if self._backward_session is not None:
            raise RuntimeError("release_graph must run outside a backward session")
        for handle in list(self._tensor_hooks.values()):
            handle.remove()
        self._tensor_hooks.clear()

    def capture_optimizer_state(self, optimizer, name="optimizer"):
        with self._suspend():
            state = optimizer.state_dict()
        with self.phase("optimizer"):
            self.save(state, name, kind="optimizer_state")

    def enable_input_replacement(self, rules, *, enabled=False):
        """Enable one-shot, explicitly targeted interventions; never infer replacement scope."""
        if enabled is not True:
            raise ValueError("input replacement requires enabled=True")
        if not isinstance(rules, list):
            raise ValueError("replacement rules must be a list")
        validated = []
        for rule in rules:
            required = {"module", "tensor_path", "step", "rank", "microbatch", "value"}
            if not isinstance(rule, dict) or set(rule) != required:
                raise ValueError("replacement requires exact module/tensor_path/step/rank/microbatch/value")
            if any(type(rule[key]) is not int or rule[key] < 0 for key in ("step", "rank", "microbatch")):
                raise ValueError("replacement coordinates must be nonnegative integers")
            if any(not isinstance(rule[key], str) or not rule[key] for key in ("module", "tensor_path")):
                raise ValueError("replacement module and tensor_path must be explicit names")
            value = rule["value"]
            if isinstance(value, (str, Path)):
                value = torch.from_numpy(np.load(value, allow_pickle=False))
            if not isinstance(value, torch.Tensor):
                raise TypeError("replacement value must be a tensor or numeric npy path")
            with self._suspend():
                value = value.detach().clone()
            validated.append({**rule, "value": value, "used": False})
        self._replacement_rules = validated

    def _replace_inputs(self, name, args, kwargs):
        rules = [r for r in self._replacement_rules if not r["used"] and r["module"] == name
                 and r["rank"] == self.rank and r["step"] == self._context["step"]
                 and r["microbatch"] == self._context["microbatch"] and self._context["phase"] != "backward"]
        if not rules:
            return args, kwargs
        if len({r["tensor_path"] for r in rules}) != len(rules):
            raise ValueError("multiple replacement rules target the same input")
        found = {}
        def replace(value, path):
            matching = next((r for r in rules if r["tensor_path"] == path), None)
            if matching is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("replacement target must be a tensor leaf")
                source = matching["value"]
                if tuple(source.shape) != tuple(value.shape) or source.dtype != value.dtype or source.layout != value.layout:
                    raise ValueError("replacement shape/dtype/layout must match the original tensor")
                replacement = source.to(value.device).clone().requires_grad_(value.requires_grad)
                found[path] = (matching, value, replacement)
                return replacement
            if type(value) in (tuple, list):
                return type(value)(replace(v, f"{path}[{i}]") for i, v in enumerate(value))
            if type(value) is dict:
                return {key: replace(v, f"{path}[{key!r}]") for key, v in value.items()}
            return value
        with self._suspend():
            new_args, new_kwargs = replace(args, "args"), replace(kwargs, "kwargs")
        if len(found) != len(rules):
            raise ValueError("replacement path missing or unsupported input container")
        for path, (rule, before, after) in found.items():
            meta = dict(self._context, module=name, call_index=0, op=None,
                        intervention={"type": "input_replacement", "gradient_semantics": "replacement_is_new_leaf"})
            self._write(before, meta, "input_replacement_before", path)
            self._write(after, meta, "input_replacement", path)
            rule["used"] = True
        return new_args, new_kwargs

    def _graph_reference(self, tensor, *, produced=False):
        identity = id(tensor)
        record = self._graph_tensors.get(identity)
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors do not expose a version counter. Every use is
            # an unknown input rather than an invented proven dependency.
            version = None
        if (produced or record is None or record["ref"]() is not tensor
                or record["version"] != version or (version is None and not produced)):
            self._graph_sequence += 1
            record = {"ref": weakref.ref(tensor, lambda ref, key=identity: self._graph_tensors.pop(key, None)
                                         if self._graph_tensors.get(key, {}).get("ref") is ref else None),
                      "tensor_id": f"tensor_{self._graph_sequence}", "producer_event_id": None, "version": version}
            self._graph_tensors[identity] = record
        return record

    def _graph_input_ids(self, values):
        tensors = []
        seen = set()
        def walk(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (tuple, list, dict)):
                if id(value) in seen:
                    return
                seen.add(id(value))
                items = value.values() if isinstance(value, dict) else value
                for item in items:
                    walk(item)
        walk(values)
        return list(dict.fromkeys(self._graph_reference(t)["tensor_id"] for t in tensors))
