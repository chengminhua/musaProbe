"""Offline tensor interchange and installed-framework evidence without auto-install."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import traceback
import numpy as np


FRAMEWORKS = ("pytorch", "mindspore", "torchair", "atb", "vllm", "sglang", "verl")
_PACKAGES = {"pytorch": "torch", "mindspore": "mindspore", "torchair": "torchair",
             "atb": "atb_llm", "vllm": "vllm", "sglang": "sglang", "verl": "verl"}


def inspect_framework(name):
    """Return local package/plugin evidence. Installation is never execution proof."""
    if name not in FRAMEWORKS: raise ValueError("Unknown framework")
    package = _PACKAGES[name]
    discovery_error = None
    try: spec = importlib.util.find_spec(package)
    except (ImportError, ValueError, OSError) as exc:
        spec = None
        discovery_error = f"{type(exc).__name__}: {exc}"
    try: version = importlib.metadata.version(package.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError: version = None
    evidence = []
    try:
        for entry in importlib.metadata.entry_points(group="vllm.platform_plugins"):
            evidence.append({"group": entry.group, "name": entry.name, "value": entry.value})
    except (TypeError, AttributeError): pass
    status = ("discovery_error" if discovery_error else "available_for_probe" if spec else
              "unavailable_import_path" if version is not None else "missing_dependency")
    return {"framework": name, "package": package, "installed": spec is not None,
            "distribution_present": version is not None,
            "version": version, "module_path": None if spec is None else spec.origin,
            "discovery_error": discovery_error,
            "platform_plugins": evidence if name == "vllm" else [],
            "backend_execution_verified": False, "status": status,
            "reason": "Package discovery does not prove a MUSA backend. Run an explicit bounded backend example; no package was installed or imported."}


def run_framework_smoke(name, model_dir, output_dir, *, device="musa"):
    """Explicit local-model inference/data-contract smoke; never downloads weights.

    vLLM/SGLang exercise one request. VeRL exercises its DataProto tensor
    contract only, not distributed RL training. The returned scope is explicit.
    """
    if name not in ("vllm", "sglang", "verl") or device not in ("musa", "cuda", "cpu"):
        raise ValueError("Unsupported framework/device")
    root=Path(output_dir).expanduser().resolve(); root.mkdir(parents=True, exist_ok=False)
    result={"framework":name,"device":device,"status":"unsupported","backend_execution_verified":False,
            "scope":"data_protocol_roundtrip" if name=="verl" else "single_local_model_request"}
    engine=None
    try:
        if name=="verl":
            import torch
            if device=="musa": import torch_musa  # noqa: F401
            from verl import DataProto
            batch=DataProto.from_dict(tensors={"input_ids":torch.tensor([[1,2,3]]),"attention_mask":torch.ones(1,3,dtype=torch.long)})
            actual=batch.batch["input_ids"].cpu().numpy()
            np.save(root/"input_ids.npy",actual,allow_pickle=False)
            result.update(status="pass" if actual.tolist()==[[1,2,3]] else "mismatch",
                          reason="CPU data adapter only; actor/rollout and distributed training were not executed")
        else:
            source=Path(model_dir).expanduser().resolve()
            if not source.is_dir() or not (source/"config.json").is_file():
                raise ValueError("An existing local HF model directory is required; downloads are disabled")
            if name=="vllm":
                from vllm import LLM, SamplingParams
                from vllm.platforms import current_platform
                observed=str(current_platform.device_type)
                result["observed_backend"]=observed
                result["platform_class"]=type(current_platform).__module__+"."+type(current_platform).__name__
                if observed!=device: raise RuntimeError(f"Requested {device}, vLLM reports {observed}")
                engine=LLM(model=str(source),tokenizer=str(source),trust_remote_code=False,
                           tensor_parallel_size=1,max_model_len=64,enforce_eager=True,
                           gpu_memory_utilization=0.1,max_num_seqs=1,max_num_batched_tokens=64)
                outputs=engine.generate([{"prompt_token_ids":[1,2,3]}],SamplingParams(temperature=0,max_tokens=4))
                ids=list(outputs[0].outputs[0].token_ids)
            else:
                import torch
                if device=="musa": import torch_musa  # noqa: F401
                if device=="cpu" or not getattr(torch,device).is_available():
                    raise RuntimeError("SGLang GPU smoke needs an available explicit GPU backend")
                from sglang import Engine
                engine=Engine(model_path=str(source),tokenizer_path=str(source),trust_remote_code=False,
                              device=device,tp_size=1,context_length=64,mem_fraction_static=0.1,
                              max_total_tokens=128,max_running_requests=1,max_prefill_tokens=64,
                              chunked_prefill_size=64,
                              disable_cuda_graph=True)
                output=engine.generate(input_ids=[1,2,3],sampling_params={"temperature":0,"max_new_tokens":4})
                ids=output.get("output_ids")
                if ids is None: raise RuntimeError("Engine output did not expose token IDs")
            np.save(root/"output_ids.npy",np.asarray(ids,dtype=np.int64),allow_pickle=False)
            result.update(status="pass" if 0<len(ids)<=4 else "mismatch",output_token_ids=ids,
                          backend_execution_verified=True)
    except Exception as exc:
        result.update(status="unsupported",reason=f"{type(exc).__name__}: {exc}")
        (root/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        if engine is not None:
            try:
                if hasattr(engine,"shutdown"):
                    engine.shutdown()
                    result["engine_cleanup"]="shutdown"
                elif name=="vllm":
                    # This function owns its engine. Target vLLM v1 exposes
                    # shutdown on EngineCoreClient, not the top-level LLM.
                    engine.llm_engine.engine_core.shutdown()
                    result["engine_cleanup"]="v1_engine_core.shutdown"
                else:
                    raise RuntimeError("Owned engine has no verified shutdown API")
            except Exception as exc: result.update(status="error",cleanup_error=f"{type(exc).__name__}: {exc}")
        (root/"framework.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def import_tensor_bundle(manifest, output_dir, *, framework=None):
    """Import explicit numeric NPY records as one standard musaProbe rank capture.

    Input format: {framework, tensors:[{name,file,shape,dtype, ...context}]}; file
    paths are relative to the manifest. No pickle, PT checkpoint or source code
    is executed. The caller supplies semantic names; no automatic mapping claim.
    """
    from .compare import _DTYPES
    path = Path(manifest).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    framework = framework or data.get("framework")
    if framework not in FRAMEWORKS: raise ValueError("Unknown source framework")
    records = data.get("tensors")
    if not isinstance(records, list) or not records or len(records) > 10000:
        raise ValueError("Expected 1..10000 tensor records")
    validated = []
    total = 0
    names = set()
    for item in records:
        name, relative = item.get("name"), item.get("file")
        if not isinstance(name, str) or not name or name in names: raise ValueError("Tensor names must be nonempty and unique")
        names.add(name)
        if not isinstance(relative, str) or Path(relative).is_absolute(): raise ValueError("Invalid tensor path")
        source = (path.parent/relative).resolve()
        if not source.is_relative_to(path.parent): raise ValueError("Tensor path escapes source directory")
        total += source.stat().st_size
        if source.stat().st_size > 64*1024**2 or total > 256*1024**2: raise ValueError("Import read budget exceeded")
        array = np.load(source, allow_pickle=False, mmap_mode="r")
        logical = item.get("dtype", str(array.dtype)).removeprefix("torch.")
        if logical not in _DTYPES or array.dtype.hasobject or _DTYPES[logical] != str(array.dtype):
            raise ValueError("Unsupported or inconsistent logical/storage dtype")
        if item.get("shape", list(array.shape)) != list(array.shape): raise ValueError("Shape mismatch")
        for key in ("step", "call_index"):
            if type(item.get(key, 0)) is not int or item.get(key, 0) < 0: raise ValueError("Invalid event context")
        validated.append((item, source, list(array.shape), logical, str(array.dtype)))
    root = Path(output_dir).expanduser().resolve()
    if root == path.parent or root.is_relative_to(path.parent):
        raise ValueError("Import output must be separate from input directory")
    root.mkdir(parents=True, exist_ok=False)
    rank = root/"rank_00000"; rank.mkdir(); (rank/"tensors").mkdir()
    events = []
    for index, (item, source, shape, logical, stored) in enumerate(validated):
        relative = f"tensors/tensor_{index:09d}.npy"
        shutil.copyfile(source, rank/relative)
        event = {"schema_version": 1, "event_id": index, "step": item.get("step", 0),
                 "microbatch": item.get("microbatch"), "phase": item.get("phase", "forward"),
                 "module": item["name"], "call_index": item.get("call_index", 0),
                 "kind": item.get("kind", "module_output"), "tensor_path": item.get("tensor_path", "$output"),
                 "shape": shape, "dtype": "torch."+logical, "stored_dtype": stored,
                 "state": "tensor", "file": relative, "device": "offline_import", "stats": {},
                 "source_framework": framework, "op": None, "recompute": False}
        events.append(event)
    (rank/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events), encoding="utf-8")
    summary = {"schema_version": 1, "tool": "musaProbe", "status": "complete", "rank": 0,
               "world_size": 1, "run_id": root.name, "events_count": len(events), "tensor_bytes": total,
               "config": {"mode": "tensor"}, "states": {"tensor": len(events)},
               "metadata": {"source_framework": framework, "import_scope": "offline_explicit_name_mapping",
                            "backend_execution_verified": False, "python": platform.python_version()}}
    (rank/"manifest.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return {"status": "complete", "rank_dir": str(rank), "events_count": len(events), "framework": framework,
            "backend_execution_verified": False}
