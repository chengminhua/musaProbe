"""Explicit external-Llama integration for one exclusive eager SGLang request.

Configure this session before importing SGLang's model registry, and construct
an exclusively owned Engine. Its public SGLANG_EXTERNAL_MODEL_PACKAGE extension
loads our Llama subclass in the actual worker. Each model forward owns a fresh
collector and closes it before returning. Control files contain only bounded
configuration; they do not execute code or expose a worker RPC endpoint.
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
import sys
import threading
import uuid

from musaprobe.config import ProbeConfig
from musaprobe.storage import atomic_json

EXTERNAL_MODEL_PACKAGE = "musaprobe.integrations.sglang_models"
_BOOTSTRAP_ENV = "MUSAPROBE_SGLANG_SESSION"


class SGLangCaptureError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _read_json(path, limit=65536):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("SGLang capture control file exceeds its read budget")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("SGLang capture control must be a JSON object")
    return value


def _configuration(values):
    values = dict(values or {})
    allowed = {"mode", "modules", "max_tensor_bytes", "max_rank_bytes", "max_events", "async_write"}
    if set(values) - allowed:
        raise ValueError("Unsupported SGLang module-capture options: " + ", ".join(sorted(set(values) - allowed)))
    values.setdefault("modules", ["model.layers.0.mlp.down_proj"])
    cfg = ProbeConfig(output_dir="unused", run_id="validation", mode=values.pop("mode", "tensor"),
                      capture_gradients=False, stack_depth=0, rank=0, world_size=1,
                      max_tensor_bytes=values.pop("max_tensor_bytes", 1024 * 1024),
                      max_rank_bytes=values.pop("max_rank_bytes", 4 * 1024 * 1024),
                      max_events=values.pop("max_events", 128), async_queue_size=2, **values)
    if cfg.mode != "tensor" or not cfg.modules or "<root>" in cfg.modules:
        raise ValueError("SGLang adapter requires tensor mode and explicit tensor-only leaf module selection")
    if not (0 < cfg.max_tensor_bytes <= 1024 * 1024 and 0 < cfg.max_rank_bytes <= 4 * 1024 * 1024 and cfg.max_events <= 128):
        raise ValueError("SGLang bounds are 1 MiB/tensor, 4 MiB/forward and 128 events/forward")
    return cfg.to_dict()


def _validate_server(args):
    required = {"tp_size": 1, "pp_size": 1, "dp_size": 1, "nnodes": 1,
                "max_running_requests": 1, "disable_cuda_graph": True,
                "disable_overlap_schedule": True, "disable_radix_cache": True,
                "enable_torch_compile": False, "speculative_algorithm": None,
                "disaggregation_mode": "null"}
    errors = [name for name, expected in required.items() if
              getattr(args, name, object()) != expected or type(getattr(args, name, None)) is not type(expected)]
    if errors:
        raise ValueError("Unsupported SGLang configuration: " + ", ".join(errors))
    return required


class _WorkerCapture:
    """Owned by one explicitly registered Llama instance, never a global hook."""

    def __init__(self, model, server_args, bootstrap_path):
        self.model = model
        self.constraints = _validate_server(server_args)
        if getattr(model.config, "model_type", None) != "llama" or getattr(model.config, "architectures", None) != ["LlamaForCausalLM"]:
            raise ValueError("External capture package supports only the explicit LlamaForCausalLM architecture")
        if getattr(model, "quant_config", None) is not None:
            raise ValueError("Quantized SGLang models are outside this adapter's validated scope")
        self.bootstrap_path = Path(bootstrap_path).resolve(strict=True)
        self.root = self.bootstrap_path.parent
        boot = _read_json(self.bootstrap_path)
        if set(boot) != {"schema_version", "config", "max_forward_calls"} or boot["schema_version"] != 1:
            raise ValueError("Invalid SGLang capture bootstrap")
        self.config = _configuration(boot["config"])
        self.maximum = boot["max_forward_calls"]
        if type(self.maximum) is not int or not 1 <= self.maximum <= 16:
            raise ValueError("max_forward_calls must be in 1..16")
        selected = [name for name, _ in model.named_modules()
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.config["modules"])]
        if not selected:
            raise ValueError("No SGLang worker module matches the capture configuration")
        self.token = None
        self.calls = 0
        self.lock = threading.Lock()
        ready = self.root / "worker_ready.json"
        if ready.exists():
            raise RuntimeError("Capture session permits one Llama worker model instance only")
        atomic_json(ready, {"status": "ready", "worker_pid": os.getpid(), "selected_modules": selected,
                    "model_class": type(model).__module__ + "." + type(model).__qualname__,
                    "constraints": self.constraints, "extension": EXTERNAL_MODEL_PACKAGE})

    def forward(self, forward_batch, invoke):
        control = _read_json(self.root / "control.json")
        if control == {"schema_version": 1, "enabled": False}:
            return invoke()
        if set(control) != {"schema_version", "enabled", "token", "request_tag"} or control.get("enabled") is not True or control["schema_version"] != 1:
            raise ValueError("Invalid active SGLang request control")
        token = control["token"]
        if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("Invalid SGLang capture ownership token")
        if not isinstance(control["request_tag"], str) or not 0 < len(control["request_tag"]) <= 256:
            raise ValueError("Invalid SGLang request tag")
        if self.token not in (None, token):
            raise RuntimeError("This SGLang adapter accepts one captured request per worker")
        self.token = token
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent SGLang model forwards are unsupported")
        probe = None
        index = self.calls
        self.calls += 1
        report_path = self.root / "forwards" / f"forward_{index:04d}.json"
        report = {"status": "pending", "hooks_removed": False, "worker_pid": os.getpid(),
                  "forward_index": index, "token": token, "request_tag": control["request_tag"]}
        try:
            atomic_json(report_path, report)
            if index >= self.maximum:
                raise RuntimeError("SGLang request exceeded its bounded forward-call budget")
            mode = getattr(getattr(forward_batch, "forward_mode", None), "name", None)
            if getattr(forward_batch, "batch_size", None) != 1 or mode not in ("EXTEND", "DECODE"):
                raise ValueError("Only batch-size-one native EXTEND/DECODE forwards are supported")
            from musaprobe import PrecisionDebugger
            configuration = dict(self.config, output_dir=str(self.root / "captures"), run_id=f"forward_{index:04d}")
            probe = PrecisionDebugger(config=configuration, metadata={
                "framework": "sglang", "integration": "SGLANG_EXTERNAL_MODEL_PACKAGE",
                "request_tag": control["request_tag"], "worker_pid": os.getpid(),
                "request_association": "caller_declared_exclusive_generate_window; native request IDs unavailable at model boundary",
                "native_forward_mode": mode, "forward_index": index, "constraints": self.constraints,
                "capture_scope": "single_Llama_worker_eager_leaf_module_forward_boundaries"})
            probe.set_context(step=index, microbatch=None, phase="forward", request_tag=control["request_tag"],
                              engine_forward_call=index, native_forward_mode=mode)
            probe.start(self.model)
            result = invoke()
        except BaseException as exc:
            report.update(status="error", reason=type(exc).__name__ + ": " + str(exc))
            if probe is not None:
                probe.errors.append("SGLang model forward failed: " + report["reason"])
            raise
        finally:
            try:
                if probe is not None:
                    probe.close()
                    report.update(rank_dir=str(probe.rank_dir), events_count=probe.writer.manifest["events_count"],
                                  states=probe.writer.manifest["states"], manifest_status=probe.writer.manifest["status"],
                                  native_forward_mode=mode)
                    report["hooks_removed"] = not probe._hooks and not probe._tensor_hooks
                    if report["status"] == "pending":
                        report["status"] = "complete" if report["manifest_status"] == "complete" else "error"
                else:
                    report["hooks_removed"] = True
            except BaseException as cleanup:
                report.update(status="error", cleanup_error=type(cleanup).__name__ + ": " + str(cleanup))
                if probe is not None:
                    try:
                        probe.stop()
                        report["hooks_removed"] = not probe._hooks and not probe._tensor_hooks
                    except BaseException:
                        report["hooks_removed"] = False
                raise
            finally:
                try:
                    atomic_json(report_path, report)
                finally:
                    self.lock.release()
        if report["status"] != "complete" or not report["hooks_removed"]:
            raise RuntimeError("SGLang forward capture did not close cleanly")
        return result


class SGLangCaptureSession:
    """Opt in before Engine construction, then capture one exclusive generate.

    Engine requirements: TP/PP/DP/nnodes=1, eager, no overlap, no radix cache,
    max_running_requests=1, no speculation/disaggregation/quantization. Use an
    isolated task process and shut down its owned Engine before closing this
    session. Concurrent callers and service-level request routing are excluded.
    """

    def __init__(self, output_dir, *, config=None, max_forward_calls=8):
        if "sglang.srt.models.registry" in sys.modules:
            raise RuntimeError("Create SGLangCaptureSession before importing SGLang's model registry")
        if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") or os.environ.get(_BOOTSTRAP_ENV):
            raise RuntimeError("An external SGLang model package/session already owns this process")
        validated = _configuration(config)
        if type(max_forward_calls) is not int or not 1 <= max_forward_calls <= 16:
            raise ValueError("max_forward_calls must be in 1..16")
        self.root = Path(output_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "forwards").mkdir()
        self.used = False
        self.closed = False
        self._request_lock = threading.Lock()
        self.maximum = max_forward_calls
        self.token = uuid.uuid4().hex
        # Store only accepted user options; runtime supplies identity/output.
        keys = ("mode", "modules", "max_tensor_bytes", "max_rank_bytes", "max_events", "async_write")
        atomic_json(self.root / "bootstrap.json", {"schema_version": 1,
                    "config": {key: validated[key] for key in keys}, "max_forward_calls": max_forward_calls})
        self._disarm()
        self.previous_env = {key: os.environ.get(key) for key in ("SGLANG_EXTERNAL_MODEL_PACKAGE", _BOOTSTRAP_ENV)}
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = EXTERNAL_MODEL_PACKAGE
        os.environ[_BOOTSTRAP_ENV] = str(self.root / "bootstrap.json")

    def _disarm(self):
        atomic_json(self.root / "control.json", {"schema_version": 1, "enabled": False})

    def generate(self, engine, *, input_ids, sampling_params, request_tag):
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent SGLang capture requests are unsupported")
        try:
            return self._generate(engine, input_ids=input_ids, sampling_params=sampling_params, request_tag=request_tag)
        finally:
            self._request_lock.release()

    def _generate(self, engine, *, input_ids, sampling_params, request_tag):
        if self.closed or self.used:
            raise RuntimeError("A session permits exactly one captured request")
        if not isinstance(input_ids, list) or not input_ids or any(type(i) is not int or i < 0 for i in input_ids):
            raise ValueError("Provide one nonempty token-ID list")
        if not isinstance(sampling_params, dict) or type(sampling_params.get("max_new_tokens")) is not int or not 1 <= sampling_params["max_new_tokens"] <= self.maximum:
            raise ValueError("max_new_tokens must be bounded by max_forward_calls")
        if not isinstance(request_tag, str) or not 0 < len(request_tag) <= 256:
            raise ValueError("request_tag must be a nonempty string of at most 256 characters")
        _validate_server(engine.server_args)
        ready = _read_json(self.root / "worker_ready.json")
        if ready.get("status") != "ready" or ready.get("extension") != EXTERNAL_MODEL_PACKAGE:
            raise RuntimeError("The real SGLang worker did not activate the external Llama adapter")
        self.used = True
        result = None
        failure = None
        report = {"request_tag": request_tag, "token": self.token, "status": "pending", "cleanup_verified": False}
        try:
            atomic_json(self.root / "control.json", {"schema_version": 1, "enabled": True,
                        "token": self.token, "request_tag": request_tag})
            result = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        except BaseException as exc:
            failure = exc
            report["request_error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            self._disarm()
        rows = []
        try:
            rows = [_read_json(path) for path in sorted((self.root / "forwards").glob("forward_*.json"))]
            report["forwards"] = rows
            report["cleanup_verified"] = bool(rows) and all(row.get("hooks_removed") is True for row in rows)
            if not rows or len(rows) > self.maximum or [r.get("forward_index") for r in rows] != list(range(len(rows))):
                raise ValueError("Missing, non-contiguous or over-budget worker capture reports")
            for row in rows:
                if row.get("token") != self.token or row.get("worker_pid") != ready["worker_pid"] or row.get("status") != "complete" or row.get("manifest_status") != "complete" or not row.get("hooks_removed"):
                    raise ValueError("A worker forward failed or has unknown capture cleanup")
                rank = Path(row["rank_dir"]).resolve()
                if not rank.is_relative_to(self.root / "captures"):
                    raise ValueError("Worker report escaped the capture directory")
                manifest = _read_json(rank / "manifest.json")
                events = [json.loads(line) for line in (rank / "events.jsonl").read_text().splitlines()]
                if manifest["status"] != "complete" or not events or len(events) != row["events_count"]:
                    raise ValueError("Worker manifest/events are incomplete")
                for event in events:
                    path = (rank / (event.get("file") or "")).resolve()
                    if event["state"] != "tensor" or not event.get("file") or not path.is_relative_to(rank) or not path.is_file():
                        raise ValueError("Worker event lacks a complete tensor payload")
            if failure is not None:
                raise ValueError("SGLang generate failed")
            report["status"] = "complete"
        except Exception as exc:
            report.update(status="error", validation_error=type(exc).__name__ + ": " + str(exc))
            if failure is None:
                failure = exc
        atomic_json(self.root / "request.json", report)
        if failure is not None:
            if not isinstance(failure, Exception):
                raise failure
            raise SGLangCaptureError("SGLang request/capture did not complete cleanly", report) from failure
        return {"output": result, "forwards": rows, "cleanup_verified": True, "request_tag": request_tag}

    def close(self):
        if self.closed:
            return
        if self._request_lock.locked():
            raise RuntimeError("Cannot close a session while its synchronous generate is active")
        self._disarm()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
