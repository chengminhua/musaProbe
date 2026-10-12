"""Explicit vLLM worker extension for an exclusive eager inference request.

No vLLM imports at module import time. Construct your own LLM with
worker_extension_cls=WORKER_EXTENSION, then call generate_with_probe. Worker
RPCs carry control data only; tensor snapshots stay in each worker's filesystem.
"""
from __future__ import annotations

import fnmatch
import threading
import uuid


WORKER_EXTENSION = "musaprobe.integrations.vllm.MusaProbeWorkerExtension"


class CaptureRPCError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _configuration(config):
    from musaprobe.config import ProbeConfig
    values = dict(config)
    values.setdefault("max_tensor_bytes", 64*1024**2)
    values.setdefault("max_rank_bytes", 256*1024**2)
    values.setdefault("max_events", 10000)
    values.setdefault("capture_gradients", False)
    cfg = ProbeConfig.from_dict(values)
    if cfg.apis or cfg.high_level_apis or cfg.dependency_graph:
        raise ValueError("vLLM RPC adapter supports module hooks only; dispatch modes are thread-local")
    if cfg.capture_fp8:
        raise ValueError("vLLM worker FP8 adapter has not been verified; capture floating module boundaries")
    if cfg.max_tensor_bytes > 64*1024**2 or cfg.max_rank_bytes > 256*1024**2 or cfg.max_events > 10000:
        raise ValueError("vLLM adapter limits are 64 MiB/tensor, 256 MiB/rank and 10000 events")
    if not cfg.max_events or not cfg.max_rank_bytes:
        raise ValueError("A nonzero capture budget is required")
    return cfg.to_dict()


class MusaProbeWorkerExtension:
    """Mixin installed through vLLM's worker_extension_cls, without library edits."""

    def musaprobe_start(self, config, token, request_tag):
        from musaprobe import PrecisionDebugger
        probe = None
        handle = None
        try:
            if getattr(self, "_musaprobe_capture_state", None) is not None:
                raise RuntimeError("A capture already owns this worker")
            config = _configuration(config)
            if getattr(getattr(self, "model_config", None), "enforce_eager", None) is not True:
                raise ValueError("Construct this engine with enforce_eager=True before attaching module hooks")
            model = self.get_model()
            selected = [name for name, _ in model.named_modules()
                        if any(fnmatch.fnmatchcase(name, pattern) for pattern in config["modules"])
                        and not any(fnmatch.fnmatchcase(name, pattern) for pattern in config["exclude_modules"])]
            if not selected: raise ValueError("No worker model module matches the capture configuration")
            probe = PrecisionDebugger(config=config, metadata={
                "framework": "vllm", "integration": "worker_extension_cls+collective_rpc",
                "capture_scope": "exclusive_single_request_eager_module_boundaries",
                "request_tag": request_tag, "native_request_ids": [],
                "engine_phase": "unknown", "phase_classification": "not_inferred_from_token_count",
            })
            state = {"probe": probe, "handle": None, "token": token, "calls": 0,
                     "request_tag": request_tag, "forward_thread": None}

            def before_forward(module, args):
                identity = threading.get_ident()
                if state["forward_thread"] not in (None, identity):
                    raise RuntimeError("Concurrent vLLM model execution is outside this adapter's scope")
                state["forward_thread"] = identity
                probe.set_context(step=state["calls"], microbatch=None, phase="forward",
                                  request_tag=request_tag, engine_phase="unknown",
                                  engine_forward_call=state["calls"])
                state["calls"] += 1

            handle = model.register_forward_pre_hook(before_forward, prepend=True)
            state["handle"] = handle
            probe.start(model)
            self._musaprobe_capture_state = state
            return {"status": "started", "rank": probe.rank, "world_size": probe.world_size,
                    "selected_modules": selected, "rank_dir": str(probe.writer.rank_dir)}
        except Exception as exc:
            if handle is not None: handle.remove()
            cleanup_error = None
            if probe is not None:
                probe.errors.append(f"vLLM capture setup failed: {type(exc).__name__}: {exc}")
                try: probe.close()
                except Exception as cleanup: cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}", "cleanup_error": cleanup_error}

    def musaprobe_stop(self, token, error=None, native_request_ids=None):
        state = getattr(self, "_musaprobe_capture_state", None)
        if state is None:
            return {"status": "not_started", "hooks_removed": True}
        if state["token"] != token:
            return {"status": "error", "hooks_removed": False, "reason": "Capture ownership token mismatch"}
        probe = state["probe"]
        try:
            state["handle"].remove()
            if error: probe.errors.append("vLLM request failed: "+str(error))
            probe.writer.manifest["metadata"]["native_request_ids"] = list(native_request_ids or [])
            probe.writer.manifest["metadata"]["model_forward_calls"] = state["calls"]
            if not state["calls"]: probe.errors.append("No worker model forward was observed")
            probe.close()
            manifest = probe.writer.manifest
            return {"status": manifest["status"], "rank": probe.rank,
                    "rank_dir": str(probe.writer.rank_dir), "hooks_removed": True,
                    "model_forward_calls": state["calls"], "events_count": manifest["events_count"],
                    "states": manifest["states"], "errors": list(probe.errors)}
        except Exception as exc:
            # stop() removes all module hooks even if persistence/close failed.
            cleanup_error = None
            try: probe.stop()
            except Exception as cleanup: cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                    "hooks_removed": cleanup_error is None, "cleanup_error": cleanup_error}
        finally:
            self._musaprobe_capture_state = None


def generate_with_probe(llm, prompt, sampling_params, *, config, request_tag, timeout=60):
    """Generate exactly one request on an exclusively owned engine and close hooks.

    The caller must not submit other work to this engine during this function.
    A unique run_id is resolved on the driver and shared with all workers.
    Return {outputs: vLLM RequestOutput list, workers: close reports, request_tag}.
    Any setup, request, worker, or cleanup error raises CaptureRPCError with the
    available worker reports; unknown RPC state is never reported as clean.
    """
    if not isinstance(request_tag, str) or not request_tag or len(request_tag) > 256:
        raise ValueError("request_tag must be a nonempty string of at most 256 characters")
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
        raise ValueError("RPC timeout must be 0..600 seconds")
    if not isinstance(prompt, (str, dict)):
        raise ValueError("Provide one text prompt or one vLLM prompt dictionary")
    if not callable(getattr(llm, "collective_rpc", None)):
        raise ValueError("This vLLM engine does not expose collective_rpc")
    config = _configuration(config)
    token = uuid.uuid4().hex
    report = {"request_tag": request_tag, "start": [], "stop": [], "cleanup_verified": False}
    outputs = None
    failure = None
    try:
        report["start"] = llm.collective_rpc("musaprobe_start", timeout=timeout,
                                            args=(config, token, request_tag))
        if not report["start"] or any(item.get("status") != "started" for item in report["start"]):
            raise RuntimeError("One or more vLLM workers could not start capture")
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        if len(outputs) != 1: raise RuntimeError("Expected exactly one vLLM request output")
    except BaseException as exc:
        failure = exc
        report["request_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        native_ids = []
        if outputs is not None:
            try: native_ids = [str(item.request_id) for item in outputs]
            except Exception as exc:
                if failure is None: failure = exc
                report["request_error"] = f"{type(exc).__name__}: {exc}"
        try:
            report["stop"] = llm.collective_rpc("musaprobe_stop", timeout=timeout,
                                               args=(token,), kwargs={"error": report.get("request_error"),
                                                                      "native_request_ids": native_ids})
            report["cleanup_verified"] = bool(report["stop"]) and all(item.get("hooks_removed") is True for item in report["stop"])
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if failure is None: failure = exc
    if failure is not None and not isinstance(failure, Exception):
        raise failure
    if (failure is not None or not report["cleanup_verified"] or
            len(report["start"]) != len(report["stop"]) or
            any(item.get("status") != "complete" or not item.get("events_count") for item in report["stop"])):
        raise CaptureRPCError("vLLM request/capture did not complete cleanly", report) from failure
    return {"outputs": outputs, "workers": report["stop"], "request_tag": request_tag,
            "cleanup_verified": True}
