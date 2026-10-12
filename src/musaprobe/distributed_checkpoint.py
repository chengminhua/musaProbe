"""Explicit conversion of ordinary PyTorch Distributed Checkpoint tensors.

Metadata has a narrow class allowlist; data chunks use weights_only=True.
MCore N-D reformulated tensors and Python object state are not inferred.
"""
from __future__ import annotations

from collections import defaultdict
import io
import itertools
import math
from pathlib import Path, PosixPath
import pickle
import zipfile

import numpy as np

from .analysis import protect_output, write_json
from .layout import validate_shape


class _MCoreSavePlan:
    """Inert representation of the audited MCore SavePlan extension.

    This class never imports Megatron or executes its constructors. The cached
    save plan is checked as metadata only; tensor reads use the global manifest.
    """
    def __setstate__(self, state):
        fields = {"items", "storage_data", "planner_data", "usable", "mcore_data"}
        if type(state) is not dict or set(state) - fields or not {"items", "mcore_data"} <= set(state):
            raise ValueError("unsupported MCoreSavePlan fields")
        if type(state["items"]) is not list or len(state["items"]) > 65536:
            raise ValueError("invalid MCoreSavePlan items")
        if state.get("storage_data") is not None:
            raise ValueError("unsupported MCoreSavePlan cached storage data")
        if state.get("planner_data") is not None and type(state["planner_data"]) is not dict:
            raise ValueError("invalid MCoreSavePlan planner data")
        if type(state.get("usable", True)) is not bool:
            raise ValueError("invalid MCoreSavePlan usable flag")
        if type(state["mcore_data"]) is not dict or any(type(k) is not str or type(v) is not dict
                                                       for k, v in state["mcore_data"].items()):
            raise ValueError("invalid MCoreSavePlan tensor metadata")
        self.__dict__.update(state)


def _metadata(path):
    try:
        import torch
        from torch.distributed.checkpoint import metadata as metadata_module
        from torch.distributed.checkpoint import planner as planner_module
        from torch.distributed.checkpoint.filesystem import _StorageInfo
    except ImportError as exc:
        raise ValueError("torch_dist conversion requires PyTorch distributed checkpoint support") from exc
    names = ("Metadata", "TensorStorageMetadata", "TensorProperties", "ChunkStorageMetadata", "MetadataIndex", "BytesStorageMetadata", "StorageMeta", "_MEM_FORMAT_ENCODING")
    allowed = {(metadata_module.__name__, name): getattr(metadata_module, name) for name in names if hasattr(metadata_module, name)}
    allowed[("torch.distributed.checkpoint.filesystem", "_StorageInfo")] = _StorageInfo
    for name in ("Size", "float16", "float32", "float64", "bfloat16", "bool", "uint8", "int8", "int16", "int32", "int64", "strided", "contiguous_format", "channels_last", "channels_last_3d"):
        allowed[("torch", name)] = getattr(torch, name)
    allowed[("builtins", "set")] = set
    allowed[("pathlib", "PosixPath")] = PosixPath
    allowed[("megatron.core.dist_checkpointing.strategies.torch", "MCoreSavePlan")] = _MCoreSavePlan
    # These PyTorch dataclasses/enums contain only declarative cached-plan data.
    # No Megatron class or checkpoint-supplied module is imported.
    for name in ("WriteItem", "WriteItemType", "TensorWriteData"):
        allowed[(planner_module.__name__, name)] = getattr(planner_module, name)
    def checked_layout(name):
        if name != "torch.strided":
            raise ValueError("only strided checkpoint layouts are supported")
        return torch.strided
    allowed[("torch.serialization", "_get_layout")] = checked_layout

    class MetadataReader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise ValueError(f"unsupported checkpoint metadata class: {module}.{name}")
            return allowed[module, name]

    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("distributed checkpoint metadata exceeds 4 MiB budget")
    with path.open("rb") as stream:
        try:
            value = MetadataReader(stream).load()
        except (pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"invalid distributed checkpoint metadata: {exc}") from exc
    if type(value) is not metadata_module.Metadata:
        raise ValueError("distributed checkpoint metadata is not Metadata")
    plans = getattr(value, "all_local_plans", None)
    if plans is not None:
        if type(plans) is not list or len(plans) > 65536:
            raise ValueError("invalid MCore cached local plans")
        global_mcore = getattr(value, "mcore_data", None)
        if type(global_mcore) is not dict:
            raise ValueError("MCore cached plans require global tensor metadata")
        for plan in plans:
            if type(plan) is not _MCoreSavePlan:
                raise ValueError("unsupported MCore cached local plan type")
            for name, info in plan.mcore_data.items():
                if global_mcore.get(name) != info:
                    raise ValueError("MCore local/global tensor metadata disagree")
            for item in plan.items:
                if type(item) is not planner_module.WriteItem or type(item.index) is not metadata_module.MetadataIndex:
                    raise ValueError("invalid MCore cached write item")
                if type(item.type) is not planner_module.WriteItemType:
                    raise ValueError("invalid MCore cached write item type")
                if item.tensor_data is not None and type(item.tensor_data) is not planner_module.TensorWriteData:
                    raise ValueError("invalid MCore cached tensor write data")
    return value


def _rectangles(shape, chunks):
    if len(chunks) > 4096:
        raise ValueError("more than 4096 chunks per tensor exceeds validation budget")
    rectangles = []
    for chunk in chunks:
        offsets, sizes = list(chunk.offsets), list(chunk.sizes)
        validate_shape(offsets)
        validate_shape(sizes)
        if len(offsets) != len(shape) or len(sizes) != len(shape) or any(a + b > dim for a, b, dim in zip(offsets, sizes, shape)):
            raise ValueError("chunk rectangle is outside global tensor shape")
        rectangle = (tuple(offsets), tuple(sizes))
        if rectangle in rectangles:
            raise ValueError("duplicate chunk rectangle")
        for other_offsets, other_sizes in rectangles:
            if all(max(a, c) < min(a + b, c + d) for a, b, c, d in zip(offsets, sizes, other_offsets, other_sizes)):
                raise ValueError("checkpoint chunk rectangles overlap")
        rectangles.append(rectangle)
    if sum(math.prod(sizes) for _, sizes in rectangles) != math.prod(shape):
        raise ValueError("checkpoint chunks do not fully cover global tensor")
    return rectangles


def convert_torch_dist(source, output, *, prefix=None, max_bytes=1073741824, max_chunk_bytes=67108864):
    """Convert ordinary tensor rectangles into full named NumPy tensors.

    Prefix selects checkpoint storage keys, not a model state-dict object path.
    This does not restore training or materialize arbitrary Python objects.
    """
    protect_output(output, (source,))
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("distributed conversion requires a new or empty output directory")
    if type(max_bytes) is not int or max_bytes < 1 or type(max_chunk_bytes) is not int or max_chunk_bytes < 1:
        raise ValueError("conversion budgets must be positive integers")
    if prefix is not None and (not isinstance(prefix, str) or not prefix):
        raise ValueError("prefix must be nonempty")
    metadata = _metadata(source / ".metadata")
    import torch
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    if not isinstance(metadata.state_dict_metadata, dict) or not isinstance(metadata.storage_data, dict):
        raise ValueError("invalid distributed checkpoint tensor/storage metadata")
    storage = defaultdict(list)
    for index, item in metadata.storage_data.items():
        if getattr(index, "offset", None) is not None:
            storage[(index.fqn, tuple(index.offset))].append(item)
    mcore = getattr(metadata, "mcore_data", {})
    if (type(mcore) is not dict or any(type(name) is not str or type(info) is not dict
                                     for name, info in mcore.items())):
        raise ValueError("invalid MCore reformulation metadata")
    output.mkdir(parents=True, exist_ok=True)
    parameters, unsupported, skipped_objects = [], [], []
    written_bytes, read_bytes = 0, 0
    dtype_map = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
                 torch.bfloat16: np.float32, torch.bool: np.bool_, torch.uint8: np.uint8,
                 torch.int8: np.int8, torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64}
    for name, spec in sorted(metadata.state_dict_metadata.items()):
        if not isinstance(name, str) or (prefix is not None and not name.startswith(prefix)):
            continue
        if not isinstance(spec, TensorStorageMetadata):
            skipped_objects.append(name)
            continue
        destination = None
        try:
            if set(mcore.get(name, {})) - {"nd_reformulated_orig_global_shape"}:
                raise ValueError("unsupported MCore tensor metadata fields")
            if mcore.get(name, {}).get("nd_reformulated_orig_global_shape") is not None:
                raise ValueError("MCore N-D reformulated tensor requires explicit model-aware restoration")
            shape = validate_shape(list(spec.size))
            if spec.properties.dtype not in dtype_map or spec.properties.layout != torch.strided:
                raise ValueError("unsupported checkpoint tensor dtype/layout")
            dtype = np.dtype(dtype_map[spec.properties.dtype])
            bytes_needed = math.prod(shape) * dtype.itemsize
            if written_bytes + bytes_needed > max_bytes:
                raise ValueError("output tensors exceed max_bytes budget")
            rectangles = _rectangles(shape, spec.chunks)
            chunks = []
            for offset, sizes in rectangles:
                if math.prod(sizes) == 0:
                    continue
                locations = storage[(name, offset)]
                if len(locations) != 1:
                    raise ValueError("tensor chunk requires exactly one storage location")
                item = locations[0]
                if getattr(item, "transform_descriptors", None):
                    raise ValueError("transformed/compressed checkpoint chunks require an explicit decoder")
                relative = Path(item.relative_path)
                target = (source / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(source):
                    raise ValueError("checkpoint chunk path escapes source")
                if (type(item.offset) is not int or type(item.length) is not int or item.offset < 0 or item.length < 1
                        or item.offset + item.length > target.stat().st_size):
                    raise ValueError("invalid chunk byte range")
                if item.length > max_chunk_bytes or read_bytes + item.length > max_bytes:
                    raise ValueError("serialized chunks exceed byte budget")
                read_bytes += item.length
                chunks.append((target, item.offset, item.length, offset, sizes))
            filename = f"tensor_{len(parameters):08d}.npy"
            destination = np.lib.format.open_memmap(output / filename, mode="w+", dtype=dtype, shape=shape)
            for path, file_offset, length, offset, sizes in chunks:
                with path.open("rb") as stream:
                    stream.seek(file_offset)
                    blob = stream.read(length)
                # Chunk serialization is a tensor only. Never disable weights_only.
                if zipfile.is_zipfile(io.BytesIO(blob)):
                    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                        if sum(info.file_size for info in archive.infolist()) > max_chunk_bytes:
                            raise ValueError("uncompressed chunk exceeds max_chunk_bytes budget")
                tensor = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != sizes or tensor.dtype != spec.properties.dtype:
                    raise ValueError("chunk tensor disagrees with declared shape/dtype")
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                selection = tuple(slice(a, a + b) for a, b in zip(offset, sizes))
                destination[selection] = tensor.numpy()
            destination.flush()
            del destination
            destination = None
            written_bytes += bytes_needed
            parameters.append({"name": name, "file": filename, "shape": list(shape), "dtype": str(spec.properties.dtype),
                               "source_layout": {"format": "torch_dist_regular_tensor", "chunks": [{"offset": list(a), "shape": list(b)} for a, b in rectangles]}})
        except (ValueError, TypeError, OSError, RuntimeError, pickle.UnpicklingError, AttributeError) as exc:
            unsupported.append({"name": name, "reason": str(exc)})
        finally:
            if destination is not None:
                del destination
    result = {"format": "musaprobe_checkpoint", "schema_version": 1, "parameters": parameters,
              "unsupported": unsupported, "skipped_non_tensor_objects": skipped_objects,
              "source_format": "torch_dist_regular_tensor", "selected_prefix": prefix,
              "status": "complete" if parameters and not unsupported else "incomplete",
              "bytes": written_bytes, "serialized_bytes_read": read_bytes,
              "scope": "Complete ordinary tensor rectangles only. Python objects and MCore N-D reformulated tensors are not restored."}
    write_json(output / "checkpoint.json", result)
    return result
