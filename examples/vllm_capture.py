"""Capture a real vLLM worker model and verify one request before/during/after.

Uses a local tiny model only. Run in an isolated process with the same musaProbe
package on the driver and worker import paths. No HTTP server is started.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback


def run(model_dir, output_dir, device="musa"):
    root=Path(output_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"):os.environ[key]="1"
    for key,leaf in (("XDG_CACHE_HOME","cache"),("TORCHINDUCTOR_CACHE_DIR","inductor"),
                     ("TRITON_CACHE_DIR","triton"),("HF_HOME","huggingface")):
        os.environ.setdefault(key,str(root/leaf))
    result={"status":"unsupported","device":device,"scope":"single_request_vllm_worker_module_capture",
            "engine_phase":"unknown","backend_execution_verified":False}
    engine=None
    try:
        source=Path(model_dir).expanduser().resolve()
        if not source.is_dir() or not (source/"config.json").is_file():raise ValueError("Local model required")
        from vllm import LLM,SamplingParams
        from vllm.platforms import current_platform
        from musaprobe.integrations.vllm import WORKER_EXTENSION,generate_with_probe,CaptureRPCError
        result["platform"]={"class":type(current_platform).__module__+"."+type(current_platform).__name__,
                            "device_type":str(current_platform.device_type),
                            "device_name":getattr(current_platform,"device_name",None)}
        if str(current_platform.device_type)!=device:
            raise RuntimeError(f"Requested {device}, vLLM selected {result['platform']}; platform plugin registration may have failed")
        engine=LLM(model=str(source),tokenizer=str(source),trust_remote_code=False,
                   tensor_parallel_size=1,max_model_len=64,enforce_eager=True,
                   gpu_memory_utilization=0.1,max_num_seqs=1,max_num_batched_tokens=64,
                   enable_prefix_caching=False,worker_extension_cls=WORKER_EXTENSION)
        prompt={"prompt_token_ids":[1,4,5]}
        sampling=SamplingParams(temperature=0,max_tokens=4)
        before=engine.generate([prompt],sampling,use_tqdm=False)
        captured=generate_with_probe(engine,prompt,sampling,
                    config={"output_dir":str(root/"captures"),"run_id":"request_0","mode":"tensor",
                            "modules":["model.layers.0.mlp.down_proj"],"max_rank_bytes":64*1024**2},
                    request_tag="tiny_request_0")
        after=engine.generate([prompt],sampling,use_tqdm=False)
        ids=lambda output:list(output[0].outputs[0].token_ids)
        baseline,during,cleaned=ids(before),ids(captured["outputs"]),ids(after)
        if not baseline or baseline!=during or baseline!=cleaned:
            raise RuntimeError("Generated token IDs differ before/during/after capture")
        result.update(status="pass",backend_execution_verified=True,output_token_ids=during,
                      transparency="identical generated token IDs before/during/after",
                      workers=captured["workers"],cleanup_verified=captured["cleanup_verified"])
    except Exception as exc:
        result.update(status="unsupported",reason=f"{type(exc).__name__}: {exc}")
        if hasattr(exc,"report"):result["capture_report"]=exc.report
        (root/"traceback.txt").write_text(traceback.format_exc())
    finally:
        if engine is not None:
            try:
                # Verified target vLLM v1 API. This example owns the engine;
                # the reusable adapter itself never shuts down a caller's LLM.
                engine.llm_engine.engine_core.shutdown()
                result["engine_cleanup"]="v1_engine_core.shutdown"
            except Exception as exc:result.update(status="error",engine_cleanup_error=f"{type(exc).__name__}: {exc}")
        (root/"vllm_capture.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("musa","cuda"),default="musa")
    args=parser.parse_args();result=run(args.model,args.output,args.device)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="pass" else 2)
