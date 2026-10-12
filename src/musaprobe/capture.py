"""Transparent, bounded PyTorch capture with optional MUSA tensors.

No device runtime is imported implicitly: an application using MUSA should
initialize torch_musa as it normally would before constructing its model.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import json
import math
import os
import sys
import threading
import traceback
import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .config import ProbeConfig
from .storage import RunWriter
from .capture_extensions import CaptureExtensionsMixin, HighLevelMode

_PHASES = {"forward", "backward", "optimizer", "unknown"}
_UNSET = object()
_GRADIENT_METADATA_FIELDS = {
    "gradient_scope", "original_parameter_shape", "parameter_flat_range",
    "gradient_stage", "gradient_source", "gradient_parallel_coordinates",
}


def _walk(value: Any, path: str, seen: set[int] | None = None):
    """Yield tensor leaves or unsupported leaves, without traversing objects."""
    if isinstance(value, torch.Tensor):
        yield path, value, None
        return
    if value is None or isinstance(value, (str, bool, int, float, complex, torch.dtype, torch.device)):
        return
    seen = set() if seen is None else seen
    if isinstance(value, (tuple, list, Mapping)):
        if id(value) in seen:
            yield path, None, "recursive container"
            return
        seen.add(id(value))
        try:
            entries = value.items() if isinstance(value, Mapping) else enumerate(value)
            for key, item in entries:
                yield from _walk(item, f"{path}[{key!r}]", seen)
        finally:
            seen.remove(id(value))
    else:
        yield path, None, f"unsupported value type: {type(value).__module__}.{type(value).__qualname__}"


def _call_tree(value: Any, path: str, seen: set[int] | None = None):
    if isinstance(value, torch.Tensor):
        return {"tensor_path": path}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"unsupported": "nonfinite_float"}
    seen = set() if seen is None else seen
    if isinstance(value, (tuple, list, Mapping)):
        if id(value) in seen:
            return {"unsupported": "recursive_container"}
        seen.add(id(value))
        try:
            if isinstance(value, Mapping):
                if any(type(key) not in (str, int, bool) for key in value):
                    return {"unsupported": "dictionary_key_type"}
                return {"type": "dict", "items": [[key, _call_tree(item, f"{path}[{key!r}]", seen)]
                                                   for key, item in value.items()]}
            return {"type": "tuple" if isinstance(value, tuple) else "list",
                    "items": [_call_tree(item, f"{path}[{index}]", seen) for index, item in enumerate(value)]}
        finally:
            seen.remove(id(value))
    return {"unsupported": f"{type(value).__module__}.{type(value).__qualname__}"}


class _APIMode(TorchDispatchMode):
    def __init__(self, debugger: "PrecisionDebugger"):
        super().__init__()
        self.debugger = debugger

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        probe = self.debugger
        op = str(func)
        if (not probe._active or probe._is_suspended() or
                not (any(fnmatch.fnmatchcase(op, pattern) for pattern in probe.config.apis)
                     or (probe.config.dependency_graph and not probe.config.apis))):
            return func(*args, **kwargs)
        metadata = probe._api_metadata(op)
        if not probe._selected(metadata):
            return func(*args, **kwargs)
        metadata["stack"] = probe._stack()
        metadata["call_spec"] = {"args": _call_tree(args, "args"), "kwargs": _call_tree(kwargs, "kwargs")}
        if probe.config.dependency_graph:
            metadata["graph_capture"] = True
            metadata["input_tensor_ids"] = probe._graph_input_ids((args, kwargs))
        # The snapshot precedes execution, including for mutating operators.
        if probe.config.capture_inputs:
            probe._capture_structure(args, "args", "api_input", metadata)
            probe._capture_structure(kwargs, "kwargs", "api_input", metadata)
        try:
            result = func(*args, **kwargs)
        except BaseException as error:
            probe._notice(metadata, "api_output", "output", "error", f"operator raised {type(error).__name__}: {error}")
            raise
        if probe.config.capture_outputs:
            if probe.config.dependency_graph:
                seen = set()
                for _, tensor, reason in _walk(result, "output"):
                    if tensor is not None and id(tensor) not in seen:
                        probe._graph_reference(tensor, produced=True)
                        seen.add(id(tensor))
            probe._capture_structure(result, "output", "api_output", metadata)
            if probe.config.dependency_graph and getattr(func._schema, "is_mutable", False):
                # PyTorch bumps the version after returning from dispatch.
                for _, tensor, reason in _walk(result, "output"):
                    record = probe._graph_tensors.get(id(tensor)) if tensor is not None else None
                    if record is not None and record["version"] is not None:
                        record["version"] += 1
        return result


class PrecisionDebugger(CaptureExtensionsMixin):
    """Observe selected tensor boundaries without modifying values or gradients.

    ``stop()`` removes all hooks and dispatch modes; call it after backward if
    gradients are required. Tensor gradient observations can refer to aliases
    and are not unique module contributions. Explicit parameter-gradient capture
    runs at the optimizer boundary and prefers Megatron's ``main_grad``.
    """

    def __init__(self, config_path=None, *, config=None, metadata=None, fp8_adapter=None):
        if config_path is not None and config is not None:
            raise ValueError("provide config_path or config, not both")
        if config_path is not None:
            self.config = ProbeConfig.from_file(config_path)
        elif isinstance(config, ProbeConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = ProbeConfig.from_dict(config)
        else:
            raise ValueError("a config path, ProbeConfig or configuration dict is required")
        self.writer = None
        self.rank = None
        self.world_size = None
        self._active = False
        self._closed = False
        self._mode = None
        self._mode_thread = None
        self._hooks = []
        self._tensor_hooks = {}
        self._gradient_registrations = 0
        self._parameter_snapshots = set()
        self._models = []
        self._context = {"step": 0, "microbatch": None, "phase": "unknown", "recompute": False,
                         "config_version": 0, "backward_id": None}
        self._counts = defaultdict(int)
        self._api_counts = defaultdict(int)
        self._local = threading.local()
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        self.metadata = dict(metadata or {})
        self.errors = []
        self.fp8_adapter = fp8_adapter
        self._init_extensions()

    @property
    def active(self):
        return self._active

    @property
    def rank_dir(self):
        return None if self.writer is None else self.writer.rank_dir

    def _frames(self):
        if not hasattr(self._local, "frames"):
            self._local.frames = []
        return self._local.frames

    def _is_suspended(self):
        return bool(getattr(self._local, "suspended", 0))

    @contextlib.contextmanager
    def _suspend(self):
        self._local.suspended = getattr(self._local, "suspended", 0) + 1
        try:
            yield
        finally:
            self._local.suspended -= 1

    def _distributed_coordinates(self):
        rank = self.config.rank
        size = self.config.world_size
        distributed = getattr(torch, "distributed", None)
        if distributed is not None and distributed.is_available() and distributed.is_initialized():
            live_rank, live_size = distributed.get_rank(), distributed.get_world_size()
            if rank is not None and rank != live_rank:
                raise ValueError("configured rank disagrees with initialized torch.distributed")
            if size is not None and size != live_size:
                raise ValueError("configured world_size disagrees with initialized torch.distributed")
            rank, size = live_rank, live_size
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if size is None:
            size = int(os.environ.get("WORLD_SIZE", "1"))
        if size < 1 or not 0 <= rank < size:
            raise ValueError("invalid distributed rank/world_size")
        if size > 1 and not self.config._run_id_explicit:
            raise ValueError("distributed capture requires shared run_id in config or MUSAPROBE_RUN_ID before constructing PrecisionDebugger")
        return rank, size

    def start(self, model=None):
        if self._closed:
            raise RuntimeError("a closed PrecisionDebugger cannot be restarted")
        if self._active:
            if model is not None and not any(model is item for item in self._models):
                if self._models:
                    raise RuntimeError("stop before attaching a different model")
                if not isinstance(model, torch.nn.Module):
                    raise TypeError("model must be a torch.nn.Module or None")
                self._models = [model]
                try:
                    self._attach_model(model)
                    if self.config.capture_parameters:
                        self._capture_parameters(model)
                except BaseException:
                    self.stop()
                    raise
            return self
        if model is not None and not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        self.rank, self.world_size = self._distributed_coordinates()
        if self.writer is None:
            self.writer = RunWriter(self.config, rank=self.rank, world_size=self.world_size, fp8_adapter=self.fp8_adapter,
                                    event_observer=self._observe_event,
                                    metadata={**self.metadata, "torch_version": torch.__version__, "collector": "module_hooks+TorchDispatchMode",
                                              "gradient_semantics": "tensor observations; aliased boundaries are not unique module contributions",
                                              "gradient_capture_policy": "one observation per forward boundary by default; explicit backward_session(retain_graph=True) retains hooks for repeated backward",
                                              "parallel_layout": {key: value for key, value in self._context.items()
                                                                  if key not in {"step", "microbatch", "phase", "recompute"}}})
        self._active = True
        self._models = [] if model is None else [model]
        try:
            if model is not None:
                self._attach_model(model)
            self._enter_modes()
            if model is not None and self.config.capture_parameters:
                self._capture_parameters(model)
        except BaseException:
            self.stop()
            raise
        return self

    def _enter_modes(self):
        if self.config.apis or self.config.dependency_graph:
            self._mode = _APIMode(self)
            self._mode_thread = threading.get_ident()
            self._mode.__enter__()
        if self.config.high_level_apis:
            self._high_mode = HighLevelMode(self)
            self._mode_thread = threading.get_ident()
            self._high_mode.__enter__()

    def _exit_modes(self):
        if (self._mode is not None or self._high_mode is not None) and threading.get_ident() != self._mode_thread:
            raise RuntimeError("stop() must run on the thread that entered the dispatch mode")
        if self._high_mode is not None:
            mode, self._high_mode = self._high_mode, None
            mode.__exit__(None, None, None)
        if self._mode is not None:
            mode, self._mode = self._mode, None
            mode.__exit__(None, None, None)
            self._mode_thread = None

    def stop(self):
        self._exit_modes()
        self._active = False
        for handle in self._hooks + list(self._tensor_hooks.values()):
            handle.remove()
        self._hooks.clear()
        self._tensor_hooks.clear()
        self._frames().clear()
        self._python_frames().clear()
        self._graph_tensors.clear()
        return self

    def close(self):
        if self._closed:
            return
        self.stop()
        if self.writer is not None:
            if self._replacement_rules:
                self.writer.manifest["metadata"]["input_replacement"] = {
                    "configured": len(self._replacement_rules),
                    "applied": sum(bool(rule["used"]) for rule in self._replacement_rules),
                    "unmatched": [{key: value for key, value in rule.items() if key not in {"value", "used"}}
                                  for rule in self._replacement_rules if not rule["used"]],
                }
            if self.errors:
                self.writer.manifest["metadata"]["collector_errors"] = list(self.errors)
            self.writer.close(status="error" if self.errors else "complete")
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None and self.writer is not None:
            self.writer.manifest["metadata"]["application_exception"] = f"{exc_type.__name__}: {exc_value}"
            self.errors.append(f"application raised {exc_type.__name__}")
        self.close()
        return False

    def step(self, step=None):
        step = self._context["step"] + 1 if step is None else step
        self.set_context(step=step)
        return step

    def set_context(self, step=None, microbatch=_UNSET, phase=None, **extra):
        if step is not None:
            if type(step) is not int or step < 0:
                raise ValueError("step must be a nonnegative integer")
            self._context["step"] = step
        if microbatch is not _UNSET:
            if microbatch is not None and (type(microbatch) is not int or microbatch < 0):
                raise ValueError("microbatch must be a nonnegative integer or None")
            self._context["microbatch"] = microbatch
        if phase is not None:
            if phase not in _PHASES:
                raise ValueError("phase must be forward, backward, optimizer or unknown")
            self._context["phase"] = phase
        reserved = {"event_id", "schema_version", "module", "call_index", "kind", "tensor_path", "op", "state", "file", "stats", "config_version", "backward_id"}
        if reserved.intersection(extra):
            raise ValueError("context cannot override event identity fields")
        if "recompute" in extra and type(extra["recompute"]) is not bool:
            raise ValueError("recompute must be true or false")
        self._context.update(extra)
        return self

    @contextlib.contextmanager
    def phase(self, name):
        old = dict(self._context)
        self.set_context(phase=name)
        try:
            yield self
        finally:
            self._context = old

    def _stack(self):
        if self.config.stack_depth == 0:
            return []
        frames = [frame for frame in traceback.extract_stack()[:-1] if frame.filename != __file__]
        return [{"file": item.filename, "line": item.lineno, "function": item.name}
                for item in frames[-self.config.stack_depth:]]

    def _selected(self, metadata, *, ignore_microbatch=False):
        if not self._active or self._is_suspended():
            return False
        if self.config.ranks is not None and self.rank not in self.config.ranks:
            return False
        if self.config.steps is not None and metadata["step"] not in self.config.steps:
            return False
        if self.config.monitor_every_n_steps is not None:
            offset = metadata["step"] - self.config.monitor_start_step
            period = self.config.monitor_every_n_steps
            if offset < 0 or offset % period or offset // period >= self.config.monitor_max_steps:
                return False
        if (not ignore_microbatch and self.config.microbatches is not None and
                metadata.get("microbatch") not in self.config.microbatches):
            return False
        module = metadata.get("module", "<root>")
        matched = (any(fnmatch.fnmatchcase(module, pattern) for pattern in self.config.modules) and
                   not any(fnmatch.fnmatchcase(module, pattern) for pattern in self.config.exclude_modules))
        if matched and self.writer is not None and self.writer.events_count >= self.config.max_events:
            # Let the writer emit its single terminal budget notice, then avoid
            # retaining hooks and call counters for data that cannot be stored.
            self._notice(metadata, "notice", "", "truncated", "max_events reached")
            return False
        return matched

    def _failure(self, error):
        detail = f"{type(error).__name__}: {error}"
        if detail not in self.errors:
            self.errors.append(detail)
            warnings.warn("musaProbe capture failed; training continues: " + detail, RuntimeWarning, stacklevel=3)

    def _notice(self, metadata, kind, path, state, reason):
        if self.writer is None:
            return
        with self._suspend():
            try:
                self.writer.write_event(dict(metadata, kind=kind, tensor_path=path, state=state, reason=reason))
            except Exception as error:
                self._failure(error)

    def _write(self, tensor, metadata, kind, path):
        if self.writer is None:
            return None
        with self._suspend():
            try:
                event = self.writer.write_tensor(tensor, **dict(metadata, kind=kind, tensor_path=path))
                self._observe_event(event)
                return event
            except Exception as error:
                self._failure(error)
                return None

    def _capture_structure(self, value, path, kind, metadata, *, gradient_kind=None, record=True):
        for tensor_path, tensor, reason in _walk(value, path):
            if reason is not None:
                if record:
                    self._notice(metadata, kind, tensor_path, "unsupported", reason)
                continue
            if record:
                tensor_metadata = metadata
                graph_record = None
                if metadata.get("graph_capture") and kind in {"api_input", "api_output"}:
                    graph_record = self._graph_reference(tensor)
                    tensor_metadata = dict(metadata, tensor_id=graph_record["tensor_id"],
                                           producer_event_id=graph_record["producer_event_id"])
                    if kind == "api_input" and graph_record["producer_event_id"] is None:
                        tensor_metadata["graph_unknown_reason"] = "producer outside observed dispatch scope or unobserved tensor mutation"
                event = self._write(tensor, tensor_metadata, kind, tensor_path)
                if graph_record is not None and kind == "api_output" and event is not None:
                    graph_record["producer_event_id"] = event["event_id"]
            if gradient_kind and self.config.capture_gradients and tensor.requires_grad:
                self._register_tensor_gradient(tensor, metadata, gradient_kind, tensor_path)

    def _register_tensor_gradient(self, tensor, metadata, kind, tensor_path):
        # Gradient hooks do not wrap tensors in views, unlike full backward hooks.
        if self.writer.events_count >= self.config.max_events:
            return
        saved = dict(metadata, phase="backward", gradient_semantics="tensor_observation")
        saved_config = self.config
        handle = None
        def hook(gradient):
            current_config = self.config
            writer_config = self.writer.config
            self.config = saved_config
            self.writer.config = dataclasses.replace(saved_config,
                max_rank_bytes=min(saved_config.max_rank_bytes, writer_config.max_rank_bytes),
                max_events=min(saved_config.max_events, writer_config.max_events),
                max_tensor_bytes=min(saved_config.max_tensor_bytes, writer_config.max_tensor_bytes))
            session = self._backward_session
            try:
                if self._selected(saved):
                    captured = dict(saved, backward_id=session["backward_id"] if session else None)
                    self._write(gradient, captured, kind, tensor_path)
            finally:
                self.config = current_config
                self.writer.config = writer_config
                # A leaf can be reused on the next training step. Leaving the
                # old hook attached would mislabel its new gradient as old data.
                if handle is not None and not (session and session["retain_graph"]):
                    handle.remove()
                    self._tensor_hooks.pop(handle.id, None)
            return None
        try:
            handle = tensor.register_hook(hook)
            self._tensor_hooks[handle.id] = handle
            self._gradient_registrations += 1
            if self._gradient_registrations % 64 == 0:
                # Handles hold weak references. Unused branches can disappear
                # without backward, so periodically discard those dead handles.
                for identity, owned in list(self._tensor_hooks.items()):
                    reference = getattr(owned, "hooks_dict_ref", None)
                    if reference is not None and reference() is None:
                        self._tensor_hooks.pop(identity, None)
        except Exception as error:
            self._notice(metadata, kind, tensor_path, "unsupported", f"cannot register tensor gradient hook: {error}")

    def _attach_model(self, model):
        # PyTorch deduplicates shared modules. Their first path is canonical and
        # invocation counters distinguish all repeated calls under that path.
        for path, module in model.named_modules():
            path = path or "<root>"
            def pre_hook(mod, args, kwargs, name=path):
                if not self._active or self._is_suspended():
                    return None
                original_args, original_kwargs = args, kwargs
                args, kwargs = self._replace_inputs(name, args, kwargs)
                metadata = dict(self._context, module=name, op=None)
                if metadata["phase"] == "unknown":
                    metadata["phase"] = "forward"
                metadata["recompute"] = bool(metadata.get("recompute") or metadata["phase"] == "backward")
                key = (metadata["step"], metadata["microbatch"], metadata["phase"], name, metadata["recompute"])
                selected = self._selected(metadata)
                metadata["call_index"] = self._counts[key] if selected else 0
                if selected:
                    self._counts[key] += 1
                    metadata["stack"] = self._stack()
                self._frames().append((mod, metadata, selected))
                if self.config.capture_parameters:
                    self._capture_parameters(mod, prefix="" if name == "<root>" else name, recurse=False)
                if selected and (self.config.capture_inputs or self.config.capture_gradients):
                    self._capture_structure(args, "args", "input", metadata, gradient_kind="grad_input", record=self.config.capture_inputs)
                    self._capture_structure(kwargs, "kwargs", "input", metadata, gradient_kind="grad_input", record=self.config.capture_inputs)
                return (args, kwargs) if args is not original_args or kwargs is not original_kwargs else None
            def post_hook(mod, args, kwargs, output):
                if not self._active or self._is_suspended():
                    return None
                frames = self._frames()
                match = next((index for index in range(len(frames) - 1, -1, -1) if frames[index][0] is mod), None)
                if match is None:
                    return None
                _, metadata, selected = frames.pop(match)
                if selected and sys.exc_info()[0] is not None:
                    error = sys.exc_info()[1]
                    if (type(error).__name__ == "_StopRecomputationError" and
                            type(error).__module__ == "torch.utils.checkpoint"):
                        # Non-reentrant checkpointing deliberately aborts the
                        # recomputed forward once its saved values are ready.
                        # No module output was produced at this boundary.
                        notes = self.writer.manifest["metadata"].setdefault("coverage_notes", [])
                        note = "Non-reentrant checkpoint early-stop boundaries have inputs but no completed recompute output."
                        if note not in notes:
                            notes.append(note)
                    else:
                        self._notice(metadata, "output", "output", "error", f"module raised {type(error).__name__}: {error}")
                elif selected and (self.config.capture_outputs or self.config.capture_gradients):
                    self._capture_structure(output, "output", "output", metadata, gradient_kind="grad_output", record=self.config.capture_outputs)
                return None
            self._hooks.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._hooks.append(module.register_forward_hook(post_hook, with_kwargs=True, always_call=True))

    def _api_metadata(self, op):
        frames = self._frames()
        metadata = dict(frames[-1][1]) if frames else dict(self._context, module="<root>", call_index=0)
        key = (metadata["step"], metadata.get("microbatch"), metadata["phase"], metadata["module"], metadata["call_index"], op)
        metadata.update(op=op, api_call_index=self._api_counts.get(key, 0))
        if self._python_frames():
            metadata["api_parent"] = dict(self._python_frames()[-1])
        if self._selected(metadata):
            self._api_counts[key] += 1
        return metadata

    def save(self, tensor, name, kind="output"):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        key = ("manual", self._context["step"], self._context["microbatch"], self._context["phase"], name, kind)
        metadata = dict(self._context, module=name, call_index=self._counts.get(key, 0), op=None, stack=self._stack())
        if self._selected(metadata):
            self._counts[key] += 1
            self._capture_structure(tensor, "value", kind, metadata)

    def _capture_parameters(self, model, *, prefix="", recurse=True):
        for name, parameter in model.named_parameters(prefix=prefix, recurse=recurse):
            module = name.rsplit(".", 1)[0] if "." in name else "<root>"
            identity = (self._context["step"], name)
            metadata = dict(self._context, microbatch=None, module=module, call_index=0, op=None, parameter_name=name)
            if identity not in self._parameter_snapshots and self._selected(metadata, ignore_microbatch=True):
                metadata["stack"] = self._stack()
                self._parameter_snapshots.add(identity)
                self._write(parameter, metadata, "parameter", name)

    def capture_parameter_grads(self, model=None, *, gradient_resolver=None):
        """Snapshot trainable parameter gradients at the caller's boundary.

        Distributed optimizers must supply a resolver that identifies their
        actual owned gradient shard. The generic main_grad/grad path represents
        a full parameter and does not establish distributed reduction semantics.
        An explicit resolver never falls back to reading a parameter's buffer.
        """
        if not self.config.capture_gradients:
            return
        if model is None:
            if len(self._models) != 1:
                raise ValueError("provide a model or start the debugger with one before capturing parameter gradients")
            model = self._models[0]
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if gradient_resolver is not None and not callable(gradient_resolver):
            raise TypeError("gradient_resolver must be callable or None")
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            module = name.rsplit(".", 1)[0] if "." in name else "<root>"
            metadata = dict(self._context, phase="optimizer", microbatch=None, module=module,
                            call_index=0, op=None, parameter_name=name, aggregation="accumulated")
            if not self._selected(metadata, ignore_microbatch=True):
                continue
            metadata["stack"] = self._stack()
            try:
                with self._suspend():
                    metadata["original_parameter_shape"] = list(parameter.shape)
                    if gradient_resolver is None:
                        main_grad = getattr(parameter, "main_grad", None)
                        gradient = main_grad if main_grad is not None else parameter.grad
                        state = "tensor" if gradient is not None else "unsupported"
                        reason = None if gradient is not None else "parameter gradient is None at capture boundary"
                        resolved_metadata = {
                            "gradient_scope": "full_parameter",
                            "parameter_flat_range": [0, parameter.numel()],
                            "gradient_source": "main_grad" if main_grad is not None else "grad",
                            "gradient_stage": "caller_boundary_unspecified",
                            "gradient_parallel_coordinates": {},
                        }
                    else:
                        resolution = gradient_resolver(name, parameter)
                        gradient, state, reason, resolved_metadata = self._validate_gradient_resolution(parameter, resolution)
                metadata.update(resolved_metadata)
            except Exception as error:
                self._notice(metadata, "parameter_grad", name, "error",
                             f"gradient resolution failed ({type(error).__name__}): {error}")
                continue
            if state != "tensor":
                self._notice(metadata, "parameter_grad", name, state, reason or "gradient resolver did not provide a capturable tensor")
            else:
                self._write(gradient, metadata, "parameter_grad", name)

    @staticmethod
    def _validate_gradient_resolution(parameter, resolution):
        if not isinstance(resolution, Mapping):
            raise ValueError("gradient resolver must return a mapping")
        unknown = set(resolution) - {"tensor", "state", "reason", "metadata"}
        if unknown:
            raise ValueError("unknown gradient resolution fields: " + ", ".join(sorted(map(str, unknown))))
        state = resolution.get("state")
        if not isinstance(state, str) or state not in {"tensor", "unsupported", "error"}:
            raise ValueError("gradient resolver state must be tensor, unsupported or error")
        reason = resolution.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("gradient resolver reason must be a string or None")
        value = resolution.get("tensor")
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("gradient resolver tensor must be a torch.Tensor or None")
        metadata = resolution.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("gradient resolver metadata must be a mapping")
        unknown = set(metadata) - _GRADIENT_METADATA_FIELDS
        if unknown:
            raise ValueError("gradient resolver cannot override metadata fields: " + ", ".join(sorted(map(str, unknown))))
        metadata = dict(metadata)
        shape = metadata.setdefault("original_parameter_shape", list(parameter.shape))
        if (not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                or shape != list(parameter.shape)):
            raise ValueError("original_parameter_shape must match the actual parameter shape")
        scope = metadata.get("gradient_scope")
        if scope is not None and scope not in {"optimizer_owned_shard", "not_owned_by_rank", "full_parameter"}:
            raise ValueError("invalid gradient_scope")
        region = metadata.get("parameter_flat_range")
        if region is not None:
            if (not isinstance(region, list) or len(region) != 2 or
                    any(type(index) is not int for index in region) or
                    not 0 <= region[0] <= region[1] <= parameter.numel()):
                raise ValueError("parameter_flat_range must be [start,end] within the parameter")
        for field in ("gradient_stage", "gradient_source"):
            if field in metadata and (not isinstance(metadata[field], str) or not metadata[field]):
                raise ValueError(f"{field} must be a nonempty string")
        if "gradient_parallel_coordinates" in metadata:
            coordinates = metadata["gradient_parallel_coordinates"]
            if not isinstance(coordinates, dict) or any(not isinstance(key, str) for key in coordinates):
                raise ValueError("gradient_parallel_coordinates must be a dictionary with string keys")
            # Reject tensors and nonfinite metadata before passing it to storage.
            json.dumps(coordinates, allow_nan=False)
        if state == "tensor" and value is None:
            state, reason = "unsupported", reason or "gradient resolver returned no tensor owned by this rank"
            metadata["gradient_scope"] = "not_owned_by_rank"
            scope = "not_owned_by_rank"
        if state == "tensor":
            if scope not in {"optimizer_owned_shard", "full_parameter"}:
                raise ValueError("a resolved tensor requires optimizer_owned_shard or full_parameter scope")
            if "gradient_stage" not in metadata or "gradient_source" not in metadata:
                raise ValueError("a resolved tensor requires explicit gradient_stage and gradient_source")
            if scope == "optimizer_owned_shard" and not metadata.get("gradient_parallel_coordinates"):
                raise ValueError("an owned shard requires nonempty gradient_parallel_coordinates")
            if region is None:
                if scope == "optimizer_owned_shard":
                    raise ValueError("an owned shard requires parameter_flat_range")
                region = metadata["parameter_flat_range"] = [0, parameter.numel()]
            if value.numel() != region[1] - region[0]:
                raise ValueError("gradient tensor numel differs from parameter_flat_range length")
            if scope == "full_parameter" and region != [0, parameter.numel()]:
                raise ValueError("full_parameter scope must cover the whole parameter")
        return value, state, reason, metadata
