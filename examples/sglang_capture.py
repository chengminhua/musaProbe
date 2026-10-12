#!/usr/bin/env python3
"""Capture an actual eager SGLang Llama worker through its external-model API.

Run in an isolated process with a pre-existing local tiny Llama model. The
adapter must be configured before importing SGLang. This example owns and shuts
down its Engine; no model files or vendor sources are edited.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from musaprobe.integrations.sglang import SGLangCaptureSession, SGLangCaptureError
from musaprobe.storage import atomic_json


def run(model_dir, output_dir):
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {"status": "error", "framework": "sglang", "device": "musa",
              "backend_execution_verified": False,
              "scope": "explicit_external_Llama_single_worker_eager_exclusive_request"}
    session = None
    engine = None
    try:
        model = Path(model_dir).expanduser().resolve(strict=True)
        config = json.loads((model / "config.json").read_text())
        if config.get("model_type") != "llama" or config.get("architectures") != ["LlamaForCausalLM"]:
            raise ValueError("Only an explicit local LlamaForCausalLM model is supported")
        if config.get("hidden_size", 10**9) > 1024 or config.get("num_hidden_layers", 10**9) > 4:
            raise ValueError("This bounded acceptance example requires a tiny local model")
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"
        for key, leaf in (("XDG_CACHE_HOME", "cache"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                          ("TRITON_CACHE_DIR", "triton"), ("HF_HOME", "huggingface")):
            os.environ.setdefault(key, str(root / leaf))
        session = SGLangCaptureSession(root / "session", config={
            "modules": ["model.layers.0.mlp.down_proj"], "max_rank_bytes": 1024 * 1024}, max_forward_calls=8)
        import torch
        import torch_musa
        if not torch.musa.is_available():
            raise RuntimeError("Explicit MUSA backend is unavailable")
        from sglang import Engine
        engine = Engine(model_path=str(model), tokenizer_path=str(model), trust_remote_code=False,
                        device="musa", tp_size=1, pp_size=1, dp_size=1, context_length=64,
                        mem_fraction_static=.1, max_total_tokens=128, max_running_requests=1,
                        max_prefill_tokens=64, chunked_prefill_size=64,
                        disable_cuda_graph=True, disable_overlap_schedule=True, disable_radix_cache=True)
        prompt = [1, 2, 3]
        sampling = {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}
        before = engine.generate(input_ids=prompt, sampling_params=sampling)
        if list((session.root / "forwards").glob("*.json")):
            raise RuntimeError("Unarmed baseline unexpectedly produced capture events")
        captured = session.generate(engine, input_ids=prompt, sampling_params=sampling, request_tag="tiny_llama_request")
        files_before = sorted(str(path.relative_to(session.root)) for path in (session.root / "captures").rglob("*"))
        after = engine.generate(input_ids=prompt, sampling_params=sampling)
        files_after = sorted(str(path.relative_to(session.root)) for path in (session.root / "captures").rglob("*"))
        token_lists = [response.get("output_ids") for response in (before, captured["output"], after)]
        if not token_lists[0] or not token_lists[0] == token_lists[1] == token_lists[2]:
            raise RuntimeError("Token IDs differ before/during/after worker capture")
        if files_before != files_after:
            raise RuntimeError("Disarmed after-request execution produced additional capture files")
        ready = json.loads((session.root / "worker_ready.json").read_text())
        if ready["worker_pid"] == os.getpid():
            raise RuntimeError("Capture did not run in the actual SGLang worker process")
        events = []
        for report in captured["forwards"]:
            rank = Path(report["rank_dir"])
            records = [json.loads(line) for line in (rank / "events.jsonl").read_text().splitlines()]
            for event in records:
                array = np.load(rank / event["file"], allow_pickle=False)
                if event["state"] != "tensor" or not str(event["device"]).startswith("musa") or not np.isfinite(array).all():
                    raise RuntimeError("Worker event is incomplete, non-MUSA, or nonfinite")
            events.extend(records)
        if {e.get("native_forward_mode") for e in events} != {"EXTEND", "DECODE"}:
            raise RuntimeError("Actual native EXTEND and DECODE captures were not both observed")
        np.save(root / "output_ids.npy", np.asarray(token_lists[1], dtype=np.int64), allow_pickle=False)
        result.update(status="pass", backend_execution_verified=True, output_token_ids=token_lists[1],
                      tokens_equal_before_during_after=True, no_capture_outside_request=True,
                      worker_pid=ready["worker_pid"], driver_pid=os.getpid(), worker_class=ready["model_class"],
                      model_forward_calls=len(captured["forwards"]), tensor_events=len(events),
                      native_forward_modes=sorted({e["native_forward_mode"] for e in events}),
                      cleanup_verified=captured["cleanup_verified"], torch_version=str(torch.__version__),
                      torch_musa_version=str(getattr(torch_musa, "__version__", "unknown")),
                      request_association="explicit exclusive control window, not native request-ID routing")
    except Exception as exc:
        result.update(status="error", reason=type(exc).__name__ + ": " + str(exc))
        if isinstance(exc, SGLangCaptureError):
            result["capture_report"] = exc.report
        (root / "traceback.txt").write_text(traceback.format_exc())
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                result["engine_cleanup"] = "Engine.shutdown"
            except Exception as exc:
                result.update(status="error", cleanup_error=type(exc).__name__ + ": " + str(exc))
        if session is not None:
            try:
                session.close()
                result["control_disabled"] = json.loads((session.root / "control.json").read_text())["enabled"] is False
            except Exception as exc:
                result.update(status="error", session_cleanup_error=type(exc).__name__ + ": " + str(exc))
        atomic_json(root / "framework.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.model, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
