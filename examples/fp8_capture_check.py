"""Validate native/MT-TE FP8 through PrecisionDebugger storage and compare_runs.

Runs only on a real MUSA device. Every output is beneath --output. Tests both
synchronous persistence and asynchronous disk writes of fixed CPU snapshots;
does not enable async_d2h. Mutations are controlled defects, then restored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def _state(value):
    """Independent bit-level observation of actual tensor/quantizer storage."""
    import torch
    if type(value) is torch.Tensor:
        fields={"data":value.detach().view(torch.uint8)}
    else:
        metadata=value.get_metadata()
        fields={k:v for k,v in metadata.items() if isinstance(v,torch.Tensor)}
        quantizer=metadata.get("quantizer")
        if quantizer is not None:
            for name in ("scale","amax"):
                item=getattr(quantizer,name,None)
                if isinstance(item,torch.Tensor):fields["quantizer_"+name]=item
    result={}
    for name,tensor in fields.items():
        array=tensor.detach().contiguous().cpu().numpy()
        result[name]={"dtype":str(array.dtype),"shape":list(array.shape),
                      "sha256":hashlib.sha256(array.tobytes()).hexdigest()}
    return result


def _verify_saved_state(rank, event, before):
    """Verify every observed storage field, including auxiliaries unused by decode.

    The independent reference is the live pre-capture ``_state`` observation.
    Equal decoded values alone cannot establish that amax or columnwise copies
    were saved faithfully. Unknown fields therefore require an explicit adapter.
    """
    import numpy as np

    aliases = {"data": "payload", "rowwise_data": "payload",
               "fp8_scale_inv": "scale_inv", "rowwise_scale_inv": "scale_inv",
               "quantizer_scale": "quantizer_scale", "quantizer_amax": "quantizer_amax",
               "data_transpose": "data_transpose", "columnwise_data": "columnwise_data",
               "columnwise_scale_inv": "columnwise_scale_inv"}
    expected = {}
    for source_name, signature in before.items():
        if source_name not in aliases:
            raise RuntimeError(f"Unmapped original FP8 tensor field: {source_name}")
        stored_name = aliases[source_name]
        if stored_name in expected:
            raise RuntimeError(f"Ambiguous original FP8 fields for {stored_name}")
        expected[stored_name] = (source_name, signature)
    if "payload" not in expected:
        raise RuntimeError("Original FP8 state has no payload field")
    auxiliary = event.get("auxiliary")
    if auxiliary is None:
        auxiliary = {}
    if not isinstance(auxiliary, dict) or "payload" in auxiliary:
        raise RuntimeError("Invalid persisted FP8 auxiliary mapping")
    files = {"payload": event.get("file"), **auxiliary}
    missing, extra = set(expected) - set(files), set(files) - set(expected)
    if missing or extra:
        raise RuntimeError(f"Persisted FP8 field mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    root = Path(rank).resolve(strict=True)
    seen_paths, verified = set(), {}
    for stored_name, (source_name, signature) in expected.items():
        reference = files[stored_name]
        if not isinstance(reference, str) or not reference:
            raise RuntimeError(f"Missing persisted FP8 file: {stored_name}")
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
            raise RuntimeError(f"Unsafe persisted FP8 file: {stored_name}")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path in seen_paths:
            raise RuntimeError(f"Unsafe or duplicate persisted FP8 file: {stored_name}")
        seen_paths.add(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if str(array.dtype) != signature["dtype"] or list(array.shape) != signature["shape"]:
            raise RuntimeError(f"Persisted FP8 shape/dtype mismatch: {source_name}")
        if array.dtype.hasobject or not array.flags.c_contiguous:
            raise RuntimeError(f"Unsupported persisted FP8 array layout: {source_name}")
        digest = hashlib.sha256()
        flat = array.reshape(-1)
        for start in range(0, flat.size, 262144):
            digest.update(flat[start:start + 262144].tobytes())
        actual = digest.hexdigest()
        if actual != signature["sha256"]:
            raise RuntimeError(f"Persisted FP8 SHA mismatch: {source_name}")
        verified[stored_name] = {"source_field": source_name, "file": reference,
                                 "dtype": str(array.dtype), "shape": list(array.shape),
                                 "sha256": actual}
    return verified


def _raw(value):
    import torch
    if type(value) is torch.Tensor:return value.detach().view(torch.uint8)
    state=value.get_metadata()
    return state["data"] if "data" in state else state["rowwise_data"]


def _flip_first_sign(value):
    import torch
    flat=_raw(value).view(-1)
    # The chosen first element is nonzero and finite, so the sign-bit change
    # changes its numerical value without introducing a NaN payload.
    flat[0].copy_(torch.tensor(int(flat[0].item()) ^ 128,dtype=torch.uint8,device=flat.device))


def _capture(value, output, run_id, *, async_write, mutate_after_snapshot=False):
    import torch
    from musaprobe import PrecisionDebugger
    from musaprobe.native_fp8 import fp8_snapshot

    class Boundary(torch.nn.Module):
        def forward(self):return value
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.boundary=Boundary()
        def forward(self):return self.boundary()

    before=_state(value)
    saved_raw=None
    probe=PrecisionDebugger(config={"output_dir":str(output),"run_id":run_id,"mode":"tensor",
                   "modules":["boundary"],"capture_inputs":False,"capture_gradients":False,
                   "capture_fp8":True,"async_write":async_write,"async_queue_size":2,
                   "max_tensor_bytes":4*1024**2,"max_rank_bytes":16*1024**2,"max_events":20},
                   fp8_adapter=fp8_snapshot,
                   metadata={"scope":"fp8_product_storage_acceptance","async_d2h":False})
    try:
        with probe:
            probe.set_context(step=0,microbatch=0,phase="forward")
            model=Model();probe.start(model);model()
            if _state(value)!=before:raise RuntimeError("Capture changed tensor, scale or amax state")
            if mutate_after_snapshot:
                saved_raw=_raw(value).clone()
                _flip_first_sign(value)
                if _state(value)==before:raise RuntimeError("Controlled post-snapshot mutation had no effect")
            probe.stop()
        if probe.errors:raise RuntimeError("Collector errors: "+str(probe.errors))
    finally:
        if saved_raw is not None:_raw(value).copy_(saved_raw)
    if _state(value)!=before:raise RuntimeError("FP8 state did not match after capture and restoration")
    directory=Path(output)/run_id
    rank=directory/"rank_00000"
    manifest=json.loads((rank/"manifest.json").read_text())
    events=[json.loads(line) for line in (rank/"events.jsonl").read_text().splitlines()]
    if manifest["status"]!="complete" or len(events)!=1 or events[0].get("state")!="tensor":
        raise RuntimeError("Expected one successfully persisted FP8 module output")
    event=events[0]
    if event.get("stored_dtype")!="uint8" or event.get("fp8",{}).get("encoding")!="raw_float8_bits":
        raise RuntimeError("Storage did not preserve the native FP8 payload")
    persisted = _verify_saved_state(rank, event, before)
    return directory,{"manifest_status":manifest["status"],"events_count":len(events),
                      "fp8_scale_mode":event["fp8"]["scale_mode"],"state_before":before,
                      "state_unchanged":True,"post_snapshot_mutation_restored":mutate_after_snapshot,
                      "persisted_state_verified":True,"persisted_fields":persisted}


def run(output_dir):
    root=Path(output_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    for key,leaf in (("XDG_CACHE_HOME","cache"),("TRITON_CACHE_DIR","triton"),
                     ("TORCHINDUCTOR_CACHE_DIR","inductor")):
        os.environ.setdefault(key,str(root/leaf))
    result={"status":"unsupported","device":"musa","scope":"PrecisionDebugger_FP8_storage_compare",
            "async_d2h_tested":False,"cases":[]}
    try:
        import torch
        import torch_musa  # noqa: F401
        from musaprobe.compare import compare_runs
        if not torch.musa.is_available():raise RuntimeError("A real MUSA device is required")
        import transformer_engine.pytorch  # noqa: F401
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
        from transformer_engine.musa.pytorch.tensor.mtfp8_tensor import MTFP8Quantizer
        source=torch.linspace(-2,2,1024,dtype=torch.float32,device="musa").reshape(8,128)
        source_before=source.cpu().clone()
        factories={
            "native":lambda:source.to(torch.float8_e4m3fn),
            "per_tensor":lambda:Float8Quantizer(torch.ones(1,device="musa"),torch.zeros(1,device="musa"),
                                                 tex.DType.kFloat8E4M3).quantize(source),
            "mt_blockwise":lambda:MTFP8Quantizer(tex.DType.kFloat8E4M3,1,128).quantize(source),
        }
        for flavor,factory in factories.items():
            value=factory();original=_state(value)
            for async_write in (False,True):
                case=root/(flavor+"_"+("async_disk" if async_write else "sync"));case.mkdir()
                baseline,before=_capture(value,case,"baseline",async_write=async_write)
                repeat,after=_capture(value,case,"repeat",async_write=async_write,
                                      mutate_after_snapshot=async_write)
                self_comparison=compare_runs(baseline,baseline,atol=0,rtol=0,output=case/"compare_self")
                normal=compare_runs(baseline,repeat,atol=0,rtol=0,output=case/"compare_repeat")
                if self_comparison["overall"]!="pass" or normal["overall"]!="pass":
                    raise RuntimeError("Stable FP8 snapshots did not compare equal")
                raw_backup=_raw(value).clone()
                try:
                    _flip_first_sign(value)
                    changed,_=_capture(value,case,"raw_changed",async_write=async_write)
                finally:_raw(value).copy_(raw_backup)
                raw_comparison=compare_runs(baseline,changed,atol=0,rtol=0,output=case/"compare_raw_changed")
                if raw_comparison["overall"]!="fail" or raw_comparison["summary"]["statuses"].get("mismatch",0)<1:
                    raise RuntimeError("Controlled raw FP8 change was not detected numerically")
                scale_result="not_applicable_native_dtype"
                if flavor!="native":
                    metadata=value.get_metadata()
                    scale=metadata["fp8_scale_inv"] if flavor=="per_tensor" else metadata["rowwise_scale_inv"]
                    saved_scale=scale.clone()
                    try:
                        scale.mul_(2)
                        changed_scale,_=_capture(value,case,"scale_changed",async_write=async_write)
                    finally:scale.copy_(saved_scale)
                    scale_comparison=compare_runs(baseline,changed_scale,atol=0,rtol=0,output=case/"compare_scale_changed")
                    if scale_comparison["overall"]!="fail" or scale_comparison["summary"]["statuses"].get("mismatch",0)<1:
                        raise RuntimeError("Controlled FP8 inverse-scale change was not detected numerically")
                    scale_result=scale_comparison["overall"]
                if _state(value)!=original or not torch.equal(source.cpu(),source_before):
                    raise RuntimeError("Original tensor/quantization state changed unexpectedly")
                result["cases"].append({"flavor":flavor,"async_write":async_write,"status":"pass",
                           "self_comparison":self_comparison["overall"],"repeat_comparison":normal["overall"],
                           "raw_change_comparison":raw_comparison["overall"],"scale_change_comparison":scale_result,
                           "baseline_storage":before,"repeat_storage":after,"original_state_unchanged":True})
        result["status"]="pass"
    except Exception as exc:
        result.update(status="error",reason=f"{type(exc).__name__}: {exc}")
        (root/"traceback.txt").write_text(traceback.format_exc())
    (root/"fp8_capture_check.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="pass" else 2)
