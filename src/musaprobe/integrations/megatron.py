"""Explicit Megatron eager training integration, with no source-tree modifications.

Model chunks have explicit virtual pipeline identities when an interleaved
schedule supplies the same model objects registered with the training step.
The schedule's current_microbatch argument is authoritative. Backward microbatches
are associated using forward output object identity, never guessed from ordering.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import weakref
from collections.abc import Mapping


def optimizer_gradient_resolver(optimizer, *, parallel_coordinates, distributed_expected=False):
    """Lazily expose exactly the gradient shard consumed by this Megatron version.

    The installed DistributedOptimizer uses private model_param_gbuf_map and
    _get_model_param_range_map APIs. Missing/ambiguous mappings are explicit
    unsupported/error records; they must never become full-buffer snapshots.
    This resolver only reads existing views and does not perform communication.
    """
    coordinates = dict(parallel_coordinates)
    leaves = None

    def flatten(current, ancestors=()):
        if id(current) in ancestors:
            raise ValueError("cycle in optimizer.chained_optimizers")
        children = getattr(current, "chained_optimizers", None)
        if children is None:
            if not getattr(current, "is_stub_optimizer", False):
                yield current
            return
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError("chained_optimizers must be a nonempty list or tuple")
        for child in children:
            yield from flatten(child, (*ancestors, id(current)))

    def resolve(name, parameter):
        nonlocal leaves
        metadata = {
            "gradient_scope": "optimizer_owned_shard" if distributed_expected else "full_parameter",
            "original_parameter_shape": list(parameter.shape),
            "parameter_flat_range": None,
            "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip" if distributed_expected else "pre_unscale_pre_clip",
            "gradient_source": "main_grad",
            "gradient_parallel_coordinates": dict(coordinates),
        }
        def result(state, reason=None, tensor=None):
            return {"tensor": tensor, "state": state, "reason": reason, "metadata": metadata}

        try:
            if leaves is None:
                leaves = list(flatten(optimizer))
            distributed = [leaf for leaf in leaves if (
                getattr(getattr(leaf, "ddp_config", None), "use_distributed_optimizer", False)
                or getattr(getattr(leaf, "config", None), "use_distributed_optimizer", False)
                or hasattr(leaf, "model_param_gbuf_map"))]
            if distributed_expected or distributed:
                metadata.update(gradient_scope="optimizer_owned_shard",
                                gradient_stage="post_reduce_scatter_pre_unscale_pre_clip")
                if not leaves:
                    return result("unsupported", "distributed optimizer has no supported non-stub leaves")
                owners = []
                for leaf in leaves:
                    mapping = getattr(leaf, "model_param_gbuf_map", None)
                    getter = getattr(leaf, "_get_model_param_range_map", None)
                    if not isinstance(mapping, Mapping) or not callable(getter):
                        return result("unsupported", "distributed optimizer lacks the validated shard mapping interface")
                    if parameter in mapping:
                        owners.append(leaf)
                if not owners:
                    metadata["gradient_scope"] = "not_owned_by_rank"
                    return result("unsupported", "not_owned_by_rank: optimizer does not consume this parameter on this rank")
                if len(owners) != 1:
                    return result("error", f"ambiguous optimizer ownership for {name}: {len(owners)} owners")
                ranges = owners[0]._get_model_param_range_map(parameter)
                if not isinstance(ranges, Mapping) or "param" not in ranges:
                    return result("error", "optimizer shard mapping has no param range")
                owned = ranges["param"]
                start, end = getattr(owned, "start", None), getattr(owned, "end", None)
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= parameter.numel()
                        or getattr(owned, "size", end - start) != end - start):
                    return result("error", f"invalid optimizer parameter range: [{start}, {end})")
                metadata["parameter_flat_range"] = [start, end]
                gradient = getattr(parameter, "main_grad", None)
                if gradient is None:
                    return result("unsupported", "owned optimizer shard has no main_grad")
                if tuple(gradient.shape) != tuple(parameter.shape):
                    return result("error", "main_grad shape disagrees with original parameter shape")
                # Exactly the view used in DistributedOptimizer._copy_model_grads_to_main_grads.
                return result("tensor", tensor=gradient.view(-1)[start:end])

            metadata["parameter_flat_range"] = [0, parameter.numel()]
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
                metadata["gradient_source"] = "grad"
            if gradient is None:
                return result("unsupported", "parameter gradient is None at optimizer boundary")
            return result("tensor", tensor=gradient)
        except Exception as error:
            return result("error", f"optimizer gradient resolver failed: {type(error).__name__}: {error}")

    return resolve


def _tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)


def _replace_first_tensor(value, transform):
    """Preserve the container and all but its first tensor; used only by injection."""
    import torch
    if isinstance(value, torch.Tensor):
        return transform(value), True
    if isinstance(value, (tuple, list)):
        children = list(value)
        for index, child in enumerate(children):
            changed, found = _replace_first_tensor(child, transform)
            if found:
                children[index] = changed
                if isinstance(value, tuple):
                    return (type(value)(*children) if hasattr(value, "_fields") else tuple(children)), True
                return children, True
    elif isinstance(value, dict):
        children = value.copy()
        for key, child in value.items():
            changed, found = _replace_first_tensor(child, transform)
            if found:
                children[key] = changed
                return children, True
    return value, False


def read_injection(path):
    """Read a deliberately explicit, opt-in fault-injection description."""
    if not path:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"rank", "step", "microbatch", "module", "kind", "value"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Injection must contain only rank/step/microbatch/module/kind/value")
    for field in ("rank", "step", "microbatch"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError(f"Injection {field} must be an explicit nonnegative integer")
    if not isinstance(value.get("module"), str) or not value["module"]:
        raise ValueError("Injection requires an explicit module glob")
    if value.get("kind") not in ("add", "nan"):
        raise ValueError("Injection kind must be add or nan")
    if value["kind"] == "add":
        import math
        if isinstance(value.get("value"), bool) or not isinstance(value.get("value"), (float, int)):
            raise ValueError("Add injection requires a finite numerical value")
        if not math.isfinite(value["value"]):
            raise ValueError("Add injection value must be finite")
    elif "value" in value:
        raise ValueError("NaN injection has no value field")
    return value


class MegatronProbe:
    """Install reversible wrappers after musa_patch has initialized Megatron."""

    def __init__(self, config_path, *, injection_path=None):
        self.config_path = str(config_path)
        self.injection = read_injection(injection_path)
        self.debugger = None
        self._originals = []
        self._global_originals = []
        self._injection_handles = []
        self._output_contexts = {}
        self._context = {"step": 0, "microbatch": None, "phase": "unknown", "recompute": False}
        self._injection_count = 0
        self._closed = False
        self._audit = None
        self._parallel_coordinates = {}
        self._model_chunks = {}
        self._capture_model = None
        self._coordinator = None
        self._hot_config_stamp = None

    def _record(self, event, **fields):
        if self._audit is not None:
            self._audit.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
            self._audit.flush()

    def _set_context(self, **fields):
        self._context.update(fields)
        if self.debugger is not None:
            self.debugger.set_context(**self._context)

    def _reject_capture_boundary(self, reason):
        # Even if a caller catches the exception, incomplete attribution must
        # not leave a manifest that looks like a clean, complete capture.
        if self.debugger is not None and reason not in self.debugger.errors:
            self.debugger.errors.append(reason)
        self._record("capture_boundary_error", reason=reason, context=dict(self._context))
        raise RuntimeError(reason)

    def _initialize(self, model, args):
        from musaprobe import PrecisionDebugger, ProbeConfig
        import torch.distributed as dist
        from megatron.core import parallel_state
        config = ProbeConfig.from_file(self.config_path)
        coordinates = {
            "tensor": parallel_state.get_tensor_model_parallel_rank(),
            "pipeline": parallel_state.get_pipeline_model_parallel_rank(),
            "data": parallel_state.get_data_parallel_rank(),
        }
        self._parallel_coordinates = dict(coordinates)
        if os.environ.get("MUSAPROBE_VALIDATE_PP_RANKS", "0") == "1":
            selected = config.ranks is None or dist.get_rank() in config.ranks
            desired = coordinates["tensor"] == 0 and coordinates["data"] == 0
            if selected != desired:
                raise RuntimeError(
                    f"Configured ranks {config.ranks} do not select DP0/TP0 of each PP stage: "
                    f"rank={dist.get_rank()}, coordinates={coordinates}")
        metadata = {
            "integration": "megatron-eager-pipeline",
            "step_semantics": "zero-based args.curr_iteration; log iteration is step + 1",
            "microbatch_semantics": "pipeline schedules.forward_step current_microbatch argument",
            "backward_association": "forward output tensor object identity",
            "parameter_gradient_point": "before optimizer.step; distributed optimizer owned shard after reduce-scatter, before unscale/clip",
            "optimizer_shard_api": "DistributedOptimizer.model_param_gbuf_map/_get_model_param_range_map; validated installed Megatron only",
            "parallelism": {
                "tensor_model_parallel_size": int(args.tensor_model_parallel_size),
                "pipeline_model_parallel_size": int(args.pipeline_model_parallel_size),
                "data_parallel_size": int(args.data_parallel_size),
                "virtual_pipeline_model_parallel_size": getattr(args, "virtual_pipeline_model_parallel_size", None),
            },
            "parallel_coordinates": coordinates,
            "fault_injection": self.injection,
            "training": {key: getattr(args, key, None) for key in (
                "num_layers", "hidden_size", "seq_length", "micro_batch_size",
                "global_batch_size", "train_samples", "lr_warmup_samples",
                "lr_decay_samples", "bf16", "fp8", "fp8_param_gather", "exit_interval",
            )},
        }
        from musaprobe.environment import capture_training_config
        metadata.update(capture_training_config(
            hyperparameters=metadata["training"], parallelism=metadata["parallelism"],
            precision={key: getattr(args, key, None) for key in ("bf16", "fp16", "fp8", "fp8_param_gather")}))
        metadata["environment_point"] = "probe_initialization_before_first_train_step"
        manifest = os.environ.get("MUSAPROBE_SOURCE_MANIFEST")
        if manifest:
            import hashlib
            source = Path(manifest)
            metadata["source_manifest"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        self.debugger = PrecisionDebugger(config=config, metadata=metadata)
        if os.environ.get("MUSAPROBE_CONTROL_GROUP", "0") == "1":
            from musaprobe.distributed import SafeBoundaryCoordinator
            self._coordinator = SafeBoundaryCoordinator(timeout_seconds=30)
        # Injection must run before the recorder's forward hook to capture its result.
        self._install_injection(model, dist.get_rank())
        self.debugger.start(model)
        rank_dir = self.debugger.writer.rank_dir
        self._audit = (rank_dir / "integration.jsonl").open("w", encoding="utf-8")
        self._record("installed", metadata=metadata)

    def _install_injection(self, model, rank):
        if self.injection is None or self.injection["rank"] != rank:
            return
        import fnmatch
        matched = [(name, module) for name, module in model.named_modules()
                   if fnmatch.fnmatchcase(name, self.injection["module"])]
        if len(matched) != 1:
            raise ValueError(f"Injection module must match exactly one instance on rank {rank}; matched {[name for name, _ in matched]}")
        name, module = matched[0]

        def inject(_module, _inputs, output):
            target = self.injection
            if (self._context["phase"] != "forward" or self._injection_count
                    or self._context["step"] != target["step"]
                    or self._context["microbatch"] != target["microbatch"]):
                return None
            # The extra operation is intentionally outside the model's reference behavior.
            def transform(tensor):
                if not tensor.is_floating_point():
                    raise TypeError("Fault injection requires a floating-point output")
                return tensor + (target["value"] if target["kind"] == "add" else float("nan"))
            changed, found = _replace_first_tensor(output, transform)
            if not found:
                raise TypeError("Fault injection module output contains no tensor")
            self._injection_count += 1
            self._record("fault_injected", module=name, context=dict(self._context), specification=target)
            return changed

        self._injection_handles.append(module.register_forward_hook(inject))

    def install(self, training=None, schedules=None):
        if self._originals:
            raise RuntimeError("MegatronProbe is already installed")
        if training is None:
            import megatron.training.training as training
        if schedules is None:
            import megatron.core.pipeline_parallel.schedules as schedules
        train_original = training.train_step
        forward_original = schedules.forward_step
        backward_original = schedules.backward_step
        train_signature = inspect.signature(train_original)
        forward_signature = inspect.signature(forward_original)
        backward_signature = inspect.signature(backward_original)
        if "current_microbatch" not in forward_signature.parameters:
            raise RuntimeError("This Megatron forward_step lacks current_microbatch; refusing guessed labels")

        @functools.wraps(train_original)
        def train_wrapper(*positional, **keyword):
            bound = train_signature.bind(*positional, **keyword)
            models, optimizer = bound.arguments["model"], bound.arguments["optimizer"]
            if not isinstance(models, (list, tuple)) or not models:
                raise ValueError("Megatron model must be a nonempty sequence of model chunks")
            args = training.get_args()
            if self.debugger is None:
                if len({id(model) for model in models}) != len(models):
                    raise ValueError("Duplicate model chunk object would make VPP identity ambiguous")
                self._model_chunks = {id(model): index for index, model in enumerate(models)}
                if len(models) == 1:
                    self._capture_model = models[0]
                else:
                    import torch
                    self._capture_model = torch.nn.ModuleDict({"chunks": torch.nn.ModuleList(models)})
                self._initialize(self._capture_model, args)
            elif list(self._model_chunks) != [id(model) for model in models]:
                self._reject_capture_boundary("Model chunk objects or their order changed after probe installation")
            hot_path = os.environ.get("MUSAPROBE_HOT_CONFIG")
            if hot_path:
                if self._coordinator is None:
                    raise RuntimeError("Hot configuration requires MUSAPROBE_CONTROL_GROUP=1 on all ranks")
                try:
                    stamp = Path(hot_path).stat().st_mtime_ns
                except OSError:
                    stamp = None
                if stamp != self._hot_config_stamp:
                    self.debugger.request_config_reload(hot_path)
                    self._hot_config_stamp = stamp
            if self.debugger.config.stop_on_nonfinite and self.debugger.world_size > 1 and self._coordinator is None:
                raise RuntimeError("Distributed nonfinite stop requires MUSAPROBE_CONTROL_GROUP=1")
            boundary = self.debugger.step_boundary(int(args.curr_iteration), coordinate=self._coordinator)
            if boundary.get("stop"):
                raise RuntimeError("musaProbe requested coordinated stop at training step boundary")
            self._set_context(step=int(args.curr_iteration), microbatch=None, phase="unknown", recompute=False,
                              virtual_pipeline=None)
            self._output_contexts.clear()
            self._record("train_step_begin", step=self._context["step"])
            original_step = optimizer.step

            @functools.wraps(original_step)
            def optimizer_step(*step_args, **step_kwargs):
                self._set_context(microbatch=None, phase="optimizer", recompute=False)
                capture_config = self.debugger.config
                # Avoid parameter traversal and private API inspection on ranks
                # and steps that cannot produce accumulated-gradient records.
                if (capture_config.capture_gradients
                        and (capture_config.ranks is None or self.debugger.rank in capture_config.ranks)
                        and (capture_config.steps is None or self._context["step"] in capture_config.steps)):
                    resolver = optimizer_gradient_resolver(
                        optimizer, parallel_coordinates=self._parallel_coordinates,
                        distributed_expected=bool(getattr(args, "use_distributed_optimizer", False)))
                    self.debugger.capture_parameter_grads(self._capture_model, gradient_resolver=resolver)
                return original_step(*step_args, **step_kwargs)

            optimizer.step = optimizer_step
            try:
                return train_original(*positional, **keyword)
            finally:
                optimizer.step = original_step
                self._record("train_step_end", step=self._context["step"], unmatched_forward_outputs=len(self._output_contexts))
                self._output_contexts.clear()
                self._set_context(microbatch=None, phase="unknown", recompute=False)

        @functools.wraps(forward_original)
        def forward_wrapper(*positional, **keyword):
            if self.debugger is None:
                return forward_original(*positional, **keyword)
            bound = forward_signature.bind(*positional, **keyword)
            microbatch = bound.arguments.get("current_microbatch")
            if microbatch is None:
                self._reject_capture_boundary("Megatron schedule did not provide current_microbatch; cannot capture accurately")
            previous = dict(self._context)
            chunk = self._model_chunks.get(id(bound.arguments.get("model")))
            if len(self._model_chunks) > 1 and chunk is None:
                self._reject_capture_boundary("Cannot identify VPP model chunk from the schedule's model argument")
            self._set_context(microbatch=int(microbatch), phase="forward", recompute=False,
                              virtual_pipeline=chunk if len(self._model_chunks) > 1 else None)
            context = dict(self._context)
            try:
                result = forward_original(*positional, **keyword)
                for tensor in _tensors(result[0]):
                    identity = id(tensor)
                    existing = self._output_contexts.get(identity)
                    if existing is not None and existing[0]() is tensor and existing[1] != context:
                        self._reject_capture_boundary("Forward output object is shared by different microbatch/chunk contexts")
                    self._output_contexts[identity] = (
                        weakref.ref(tensor, lambda _ref, key=identity: self._output_contexts.pop(key, None)), context)
                return result
            finally:
                self._set_context(**previous)

        @functools.wraps(backward_original)
        def backward_wrapper(*positional, **keyword):
            if self.debugger is None:
                return backward_original(*positional, **keyword)
            bound = backward_signature.bind(*positional, **keyword)
            contexts = []
            seen = set()
            unknown = False
            for tensor in _tensors(bound.arguments["output_tensor"]):
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                entry = self._output_contexts.pop(id(tensor), None)
                if entry is not None and entry[0]() is tensor:
                    contexts.append(entry[1])
                else:
                    unknown = True
            previous = dict(self._context)
            if contexts and not unknown and all(item == contexts[0] for item in contexts):
                self._set_context(**{**contexts[0], "phase": "backward", "recompute": False})
            else:
                # Fail closed: do not attribute an unknown backward to the last forward.
                self._record("backward_context_unmatched", step=self._context["step"])
                self._reject_capture_boundary("Cannot associate backward output with its forward microbatch")
            try:
                with self.debugger.backward_session():
                    return backward_original(*positional, **keyword)
            finally:
                self._set_context(**previous)

        for module, name, replacement in ((training, "train_step", train_wrapper),
                                           (schedules, "forward_step", forward_wrapper),
                                           (schedules, "backward_step", backward_wrapper)):
            self._originals.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)
        # musa_patch replaces training.train with a function defined in its own
        # module. Its global train_step reference is distinct from training's
        # attribute, even though both initially point to the same function.
        caller_globals = getattr(getattr(training, "train", None), "__globals__", {})
        if caller_globals.get("train_step") is train_original:
            self._global_originals.append((caller_globals, "train_step", train_original))
            caller_globals["train_step"] = train_wrapper
        return self

    def stop(self):
        """Remove observation before distributed shutdown, keeping the writer open."""
        for namespace, name, original in reversed(self._global_originals):
            namespace[name] = original
        self._global_originals.clear()
        for module, name, original in reversed(self._originals):
            setattr(module, name, original)
        self._originals.clear()
        for handle in self._injection_handles:
            handle.remove()
        self._injection_handles.clear()
        if self.debugger is not None:
            self.debugger.stop()

    def close(self, status="complete"):
        if self._closed:
            return
        self._closed = True
        self.stop()
        try:
            if self.debugger is not None:
                if status != "complete":
                    detail = f"Megatron integration exited with status={status}"
                    if detail not in self.debugger.errors:
                        self.debugger.errors.append(detail)
                if self.debugger.errors:
                    status = "error"
                # The collector must append its errors before it closes the
                # writer. Pre-closing the writer would lose those diagnostics.
                self.debugger.close()
        except BaseException:
            status = "error"
            raise
        finally:
            if self._coordinator is not None:
                self._coordinator.close()
                self._coordinator = None
            if self._audit is not None:
                self._record("closed", status=status, injection_count=self._injection_count)
                self._audit.close()
