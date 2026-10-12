"""Explicit, non-pickle FP8 bit-pattern bundles with quantization metadata.

Native torch float8 has no external scale. TE adapters must supply actual
quantization state; a floating module boundary is not a native FP8 capture.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np


FORMATS = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")


@lru_cache(maxsize=16)
def _te_source_fingerprint(cls):
    import inspect
    import importlib.metadata
    try: version=importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError: version=None
    path=inspect.getsourcefile(cls)
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    return {"distribution_version":version,"tensor_class_source_sha256":digest}


def fp8_snapshot(tensor):
    """Storage adapter for native float8 and explicitly matched MT-TE tensor APIs."""
    import torch
    if type(tensor) is not torch.Tensor or str(tensor.dtype).removeprefix("torch.") not in FORMATS:
        return te_fp8_snapshot(tensor)
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise TypeError("Only materialized dense FP8 tensors are supported")
    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).cpu().numpy().copy()
    return {"payload": raw, "metadata": {"format": str(tensor.dtype),
            "encoding": "raw_float8_bits", "logical_shape": list(tensor.shape),
            "scale_mode": "unscaled_native_dtype", "source_type": "torch.Tensor"},
            "auxiliary": {}, "analysis_tensor": value.float()}


def _decode_arrays(info, arrays):
    import torch
    name = info["format"].removeprefix("torch.")
    if name not in FORMATS: raise ValueError("Unsupported FP8 encoding")
    raw = torch.from_numpy(arrays["payload"].copy()).view(getattr(torch, name)).float()
    mode = info["scale_mode"]
    if mode == "unscaled_native_dtype": return raw
    scale = torch.from_numpy(arrays["scale_inv"].copy()).float()
    if mode == "per_tensor":
        if scale.numel()!=1: raise ValueError("Per-tensor inverse scale must be scalar")
        return raw*scale.reshape(())
    if mode == "mt_blockwise":
        bm,bn = info["block_m"],info["block_n"]
        if type(bm) is not int or type(bn) is not int or bm<1 or bn<1 or raw.ndim<1:
            raise ValueError("Invalid blockwise scale layout")
        cols=raw.shape[-1]; rows=raw.numel()//cols if cols else 0
        if list(scale.shape)!=[(rows+bm-1)//bm,(cols+bn-1)//bn]:
            raise ValueError("Unrecognized MT-TE scale layout; refusing to infer padding")
        expanded=scale.repeat_interleave(bm,0).repeat_interleave(bn,1)[:rows,:cols]
        return (raw.reshape(rows,cols)*expanded).reshape(raw.shape)
    raise ValueError("Unsupported scale mode")


def te_fp8_snapshot(tensor):
    """Read the target MT-TE get_metadata contract without invoking a quantizer.

    Verified against the isolated target's Float8TensorBase.get_metadata and
    MTFP8TensorBase.get_metadata APIs. Unknown layouts are rejected. Snapshotting
    does not call prepare_for_saving, update_usage or _get_quantizer.
    """
    import torch
    cls=type(tensor)
    supported={
        ("transformer_engine.pytorch.tensor.float8_tensor","Float8Tensor"):"per_tensor",
        ("transformer_engine.musa.pytorch.tensor.mtfp8_tensor","MTFP8Tensor"):"mt_blockwise",
    }
    mode=supported.get((cls.__module__,cls.__name__))
    if mode is None: raise TypeError("Unknown FP8 tensor type; a supported native/MT-TE tensor is required")
    state=tensor.get_metadata()
    if not isinstance(state,dict): raise TypeError("TE metadata must be a dictionary")
    dtype_name=getattr(state["fp8_dtype"],"name",str(state["fp8_dtype"]).rsplit(".",1)[-1])
    formats={"kFloat8E4M3":"torch.float8_e4m3fn","kFloat8E5M2":"torch.float8_e5m2"}
    if dtype_name not in formats: raise ValueError("Unknown TE FP8 format")
    raw=state.get("data" if mode=="per_tensor" else "rowwise_data")
    scale=state.get("fp8_scale_inv" if mode=="per_tensor" else "rowwise_scale_inv")
    if not isinstance(raw,torch.Tensor) or raw.dtype!=torch.uint8 or not isinstance(scale,torch.Tensor):
        raise ValueError("Rowwise raw data and inverse scales are required; columnwise-only tensors are unsupported")
    if raw.numel()!=tensor.numel() or raw.numel()>64*1024**2 or scale.numel()*scale.element_size()>64*1024**2:
        raise ValueError("TE raw payload shape/budget mismatch")
    payload=raw.detach().contiguous().cpu().numpy().reshape(tuple(tensor.shape)).copy()
    auxiliary={"scale_inv":scale.detach().float().contiguous().cpu().numpy().copy()}
    info={"format":formats[dtype_name],"encoding":"raw_float8_bits","logical_shape":list(tensor.shape),
          "scale_mode":mode,"source_type":cls.__module__+"."+cls.__name__,
          "nominal_dtype":str(tensor.dtype),"te_dtype":dtype_name,"adapter_contract":"get_metadata_v1",
          "training_state_restored":False,**_te_source_fingerprint(cls)}
    q=state.get("quantizer")
    if mode=="mt_blockwise":
        if q is None: raise ValueError("Block shape requires the actual attached MT-TE quantizer")
        info.update(block_m=q.block_m,block_n=q.block_n)
        extra=("columnwise_data","columnwise_scale_inv")
    else:
        extra=("data_transpose",)
        if q is not None:
            for field in ("scale","amax"):
                value=getattr(q,field,None)
                if isinstance(value,torch.Tensor): auxiliary["quantizer_"+field]=value.detach().float().contiguous().cpu().numpy().copy()
    for field in extra:
        value=state.get(field)
        if value is not None:
            if not isinstance(value,torch.Tensor): raise TypeError("TE auxiliary state must be tensor or None")
            if value.numel()*value.element_size()>64*1024**2: raise ValueError("TE auxiliary state exceeds snapshot budget")
            auxiliary[field]=value.detach().contiguous().cpu().numpy().copy()
    analysis=_decode_arrays(info,{"payload":payload,**auxiliary})
    return {"payload":payload,"metadata":info,"auxiliary":auxiliary,"analysis_tensor":analysis}


def export_fp8_bundle(value, output_dir, *, metadata=None, adapter=fp8_snapshot):
    """Persist a checked adapter snapshot; never overwrite an existing bundle."""
    snapshot = adapter(value)
    raw = np.asarray(snapshot["payload"])
    info = dict(snapshot["metadata"])
    if raw.dtype != np.uint8 or not raw.flags.c_contiguous:
        raise ValueError("FP8 payload must be contiguous uint8")
    if info.get("format", "").removeprefix("torch.") not in FORMATS:
        raise ValueError("Unsupported FP8 format")
    shape = info.get("logical_shape")
    if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape) or int(np.prod(shape, dtype=object)) != raw.size:
        raise ValueError("FP8 payload and logical shape disagree")
    if raw.nbytes > 64 * 1024**2:
        raise ValueError("FP8 bundle payload exceeds 64 MiB")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "native_fp8_bundle", "status": "error",
                "quantization": info, "metadata": metadata or {}, "files": {}}
    try:
        if "payload" in snapshot.get("auxiliary", {}): raise ValueError("Auxiliary state cannot replace the raw payload")
        arrays = {"payload": raw, **snapshot.get("auxiliary", {})}
        total = 0
        for index, (name, array) in enumerate(arrays.items()):
            array = np.asarray(array)
            if array.dtype.hasobject or array.dtype.kind not in "buif":
                raise ValueError("Only numeric non-pickle arrays are accepted")
            total += array.nbytes
            if total > 128 * 1024**2: raise ValueError("FP8 bundle exceeds 128 MiB")
            filename = f"array_{index}.npy"
            np.save(directory / filename, array, allow_pickle=False)
            manifest["files"][name] = {"file": filename, "dtype": str(array.dtype), "shape": list(array.shape),
                                        "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest()}
        manifest["status"] = "complete"
    finally:
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {"status": "complete", "directory": str(directory), "manifest": manifest}


def load_fp8_bundle(path):
    """Load checked raw arrays on CPU; this never constructs or executes a TE object."""
    root = Path(path).expanduser().resolve()
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "native_fp8_bundle" or data.get("status") != "complete":
        raise ValueError("Incomplete or unsupported FP8 bundle")
    arrays = {}
    total = 0
    for name, spec in data["files"].items():
        relative = spec["file"]
        target = (root / relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(root):
            raise ValueError("Bundle path escapes root")
        total += target.stat().st_size
        if total > 128 * 1024**2: raise ValueError("Bundle exceeds read budget")
        if hashlib.sha256(target.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("FP8 array checksum mismatch")
        array = np.load(target, allow_pickle=False)
        if array.dtype.hasobject or str(array.dtype) != spec["dtype"] or list(array.shape) != spec["shape"]:
            raise ValueError("FP8 array schema mismatch")
        arrays[name] = array
    raw = arrays.get("payload")
    info = data["quantization"]
    if raw is None or raw.dtype != np.uint8 or list(raw.shape) != info["logical_shape"]:
        raise ValueError("Invalid raw FP8 payload")
    return {"metadata": data["metadata"], "quantization": info, "arrays": arrays}


def decode_fp8_bundle(path):
    """Decode supported native/per-tensor/MT block layouts on CPU without TE."""
    bundle = load_fp8_bundle(path)
    info = bundle["quantization"]
    return _decode_arrays(info,bundle["arrays"])
