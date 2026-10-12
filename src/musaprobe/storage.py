"""Versioned, bounded tensor snapshots. No execution of serialized objects."""
from __future__ import annotations

import json
import io
import copy
import math
import os
import platform
import threading
import queue
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .async_transfer import AsyncD2HSubmissionError

_TE_FP8_TYPES = {
    ("transformer_engine.pytorch.tensor.float8_tensor", "Float8Tensor"),
    ("transformer_engine.musa.pytorch.tensor.mtfp8_tensor", "MTFP8Tensor"),
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(_finite_json(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


class RunWriter:
    def __init__(self, config, rank=0, world_size=1, metadata=None, fp8_adapter=None, event_observer=None):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.rank_dir = Path(config.output_dir).expanduser().resolve() / config.run_id / f"rank_{rank:05d}"
        self.rank_dir.mkdir(parents=True, exist_ok=False)
        (self.rank_dir / "tensors").mkdir()
        self.events_count = 0
        self.bytes_written = 0
        self._counts = Counter()
        self._closed = False
        self._reserved_bytes = 0
        self._io_errors = []
        self.fp8_adapter = fp8_adapter
        self.event_observer = event_observer
        self._d2h = None
        if getattr(config, "async_d2h", False):
            from .async_transfer import MusaTransferPool
            self._d2h = MusaTransferPool(config.async_queue_size)
        self._event_limit_recorded = False
        self._lock = threading.RLock()
        self._stream = (self.rank_dir / "events.jsonl").open("w", encoding="utf-8")
        self.manifest = {
            "schema_version": 2, "tool": "musaProbe", "version": __version__,
            "run_id": config.run_id, "rank": rank, "world_size": world_size,
            "status": "running", "started_at": _now(),
            "python": platform.python_version(), "machine": platform.machine(),
            "config": config.to_dict(), "metadata": metadata or {},
            "events_count": 0, "tensor_bytes": 0, "states": {},
        }
        self._queue = None
        self._worker = None
        if getattr(config, "async_write", False):
            self._queue = queue.Queue(maxsize=config.async_queue_size)
            self.manifest["metadata"]["storage_policy"] = "synchronous_cpu_snapshot_async_disk_only"
            if self._d2h is not None:
                self.manifest["metadata"].update(storage_policy="musa_fixed_clone_event_d2h_async_disk",
                    async_d2h_scope="plain_dense_musa_tensors_current_producer_stream; CPU stats after DMA; fixed capacity through disk persistence",
                    async_d2h_capacity=config.async_queue_size)
            self._worker = threading.Thread(target=self._consume, name="musaprobe-writer", daemon=True)
        atomic_json(self.rank_dir / "manifest.json", self.manifest)
        if self._worker is not None:
            self._worker.start()

    def _base_event(self, metadata):
        event = {
            "schema_version": 2, "event_id": self.events_count,
            "step": 0, "microbatch": None, "phase": "unknown", "module": "",
            "call_index": 0, "kind": "output", "tensor_path": "$", "op": None,
            "shape": [], "dtype": None, "device": None, "stride": [],
            "stats": {}, "state": "unsupported", "file": None, "reason": None,
            "stack": [], "recompute": False, "config_version": 0, "backward_id": None,
        }
        event.update(metadata)
        event["event_id"] = self.events_count
        event["schema_version"] = 2
        return event

    def _append(self, event, payloads=None):
        self.events_count += 1
        if self._queue is None:
            self._persist(event, payloads or {})
        else:
            # Arrays have already been serialized into immutable CPU bytes.
            # Backpressure never discards a snapshot silently.
            self._queue.put((copy.deepcopy(event), payloads or {}))
        return event

    def _write_payload(self, relative, payload):
        target = self.rank_dir / relative
        tmp = target.with_suffix(".npy.tmp")
        try:
            with tmp.open("wb") as stream:
                stream.write(payload)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _persist(self, event, payloads):
        completed = []
        try:
            for relative, payload in payloads.items():
                self._write_payload(relative, payload)
                completed.append(relative)
            self.bytes_written += sum(len(value) for value in payloads.values())
        except Exception as exc:
            for relative in completed:
                (self.rank_dir / relative).unlink(missing_ok=True)
            event.update(state="error", file=None, reason=f"snapshot persistence failed: {type(exc).__name__}: {exc}")
            event.pop("auxiliary", None)
            self._io_errors.append(event["reason"])
        try:
            self._stream.write(json.dumps(_finite_json(event), ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()
            self._counts[event.get("state", "unsupported")] += 1
        except Exception as exc:
            self._io_errors.append(f"event persistence failed: {type(exc).__name__}: {exc}")

    def _consume(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                event, payload = item
                if isinstance(payload, dict):
                    self._persist(event, payload)
                else:
                    self._complete_d2h(event, payload)
            except Exception as exc:
                self._io_errors.append(f"writer worker failed: {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _complete_d2h(self, event, job):
        payloads = {}
        try:
            cpu = job.wait()
            event["stats"] = self._stats(cpu)
            event["state"] = "statistics"
            if event.pop("_save_tensor"):
                import torch
                array = (cpu.float() if cpu.dtype == torch.bfloat16 else cpu).contiguous().numpy()
                relative = f"tensors/tensor_{event['event_id']:09d}.npy"
                payloads = {relative: self._serialize(array)}
                actual = len(payloads[relative])
                if actual > event["_reserved_snapshot_bytes"]:
                    raise RuntimeError("async_d2h serialized payload exceeded its reserved byte budget")
                event.update(state="tensor", file=relative, stored_dtype=str(array.dtype), bytes=actual)
        except Exception as exc:
            reason = f"async_d2h snapshot failed: {type(exc).__name__}: {exc}"
            self._io_errors.append(reason)
            event.update(state="error", reason=reason, file=None)
            payloads = {}
        finally:
            event.pop("_save_tensor", None)
            event.pop("_reserved_snapshot_bytes", None)
            event.pop("_observation_deferred", None)
        try:
            self._persist(event, payloads)
            if self.event_observer is not None:
                self.event_observer(copy.deepcopy(event))
        finally:
            # A pinned buffer cannot be reused until serialization and writing
            # have completed, including the error path.
            job.release()

    def _queue_d2h(self, tensor, event):
        import torch
        predicted = tensor.numel() * (4 if tensor.dtype == torch.bfloat16 else tensor.element_size())
        reserve = predicted + 4096 if self.config.mode == "tensor" else 0
        if predicted > self.config.max_tensor_bytes:
            event.update(state="truncated", reason="max_tensor_bytes reached")
            return self._append(event)
        if self._reserved_bytes + reserve > self.config.max_rank_bytes:
            event.update(state="truncated", reason="max_rank_bytes reached (includes conservative async header reservation)")
            return self._append(event)
        pending = dict(state="pending", snapshot_transport="musa_event_d2h", stats_device="cpu",
                       _save_tensor=self.config.mode == "tensor", _reserved_snapshot_bytes=reserve,
                       _observation_deferred=True)
        # Prepare fallible Python metadata before enqueuing any device work.
        queued_event = copy.deepcopy({**event, **pending})
        job = self._d2h.capture(tensor)
        event.update(pending)
        self._reserved_bytes += reserve
        self.events_count += 1
        try:
            self._queue.put((queued_event, job))
        except BaseException as error:
            reason = f"async_d2h queue handoff failed: {type(error).__name__}: {error}"
            self._io_errors.append(reason)
            self._d2h.abandon(job, reason)
            if isinstance(error, Exception):
                raise AsyncD2HSubmissionError(reason) from error
            raise
        return event

    def flush(self):
        if self._queue is not None:
            self._queue.join()
        if self._d2h is not None:
            self._io_errors.extend(self._d2h.drain_abandoned())
        if not self._closed:
            try:
                self._stream.flush()
            except Exception as exc:
                self._io_errors.append(f"event flush failed: {type(exc).__name__}: {exc}")
        return not self._io_errors

    def _has_capacity(self):
        if self._closed:
            raise RuntimeError("musaProbe writer is closed")
        if self.events_count < self.config.max_events:
            return True
        if not self._event_limit_recorded:
            self._event_limit_recorded = True
            # A single terminal notice is allowed in addition to the configured data-event budget.
            self._append(self._base_event({"state": "truncated", "kind": "notice",
                                           "reason": "max_events reached"}))
        return False

    def write_event(self, event):
        with self._lock:
            if not self._has_capacity():
                return None
            return self._append(self._base_event(event))

    @staticmethod
    def _stats(tensor):
        import torch
        value = tensor.detach()
        numel = value.numel()
        stats = {"numel": numel, "finite_count": 0, "nan_count": 0,
                 "posinf_count": 0, "neginf_count": 0,
                 "min": None, "max": None, "mean": None, "l2": None}
        if not numel:
            return stats
        if value.is_floating_point():
            finite = torch.isfinite(value)
            stats.update(finite_count=int(finite.sum().item()),
                         nan_count=int(torch.isnan(value).sum().item()),
                         posinf_count=int(torch.isposinf(value).sum().item()),
                         neginf_count=int(torch.isneginf(value).sum().item()))
            if not stats["finite_count"]:
                return stats
            # Keep double inputs in double; lower precision uses device float32.
            # Normalize before reductions so finite FP32 inputs do not overflow
            # merely because their sum or sum of squares is large.
            work = value if value.dtype == torch.float64 else value.float()
            safe = torch.where(finite, work, torch.zeros_like(work))
            minimum = torch.where(finite, work, torch.full_like(work, float("inf"))).min().item()
            maximum = torch.where(finite, work, torch.full_like(work, -float("inf"))).max().item()
            scale = max(abs(minimum), abs(maximum))
            if scale:
                normalized = safe / scale
                mean = (normalized.sum().item() / stats["finite_count"]) * scale
                l2 = torch.linalg.vector_norm(normalized).item() * scale
            else:
                mean = l2 = 0.0
        else:
            stats["finite_count"] = numel
            # Native min/max preserve integer values; mean/norm are descriptive float32 statistics.
            minimum, maximum = value.min().item(), value.max().item()
            work = value.float()
            mean, l2 = work.mean().item(), torch.linalg.vector_norm(work).item()
        stats.update(min=minimum, max=maximum, mean=mean, l2=l2)
        return _finite_json(stats)

    def _slice(self, tensor, event):
        import fnmatch
        matches = [rule for rule in getattr(self.config, "tensor_slices", [])
                   if all(str(event.get(key, "")) == rule[key] or fnmatch.fnmatchcase(str(event.get(key, "")), rule[key])
                          for key in ("module", "kind", "tensor_path"))]
        if len(matches) > 1:
            raise ValueError("multiple slice rules match the same tensor")
        if not matches:
            return tensor
        if (type(tensor).__module__, type(tensor).__name__) in _TE_FP8_TYPES:
            raise NotImplementedError("TE raw FP8 slicing requires a scale-layout-aware adapter")
        selection = matches[0]["selection"]
        if len(selection) != tensor.ndim:
            raise ValueError("slice selection must specify every tensor dimension")
        axes = [slice(*axis).indices(size) for axis, size in zip(selection, tensor.shape)]
        event["tensor_slice"] = {"original_shape": list(tensor.shape),
                                 "selection": [dict(zip(("start", "stop", "step"), axis)) for axis in axes]}
        return tensor[tuple(slice(*axis) for axis in axes)]

    @staticmethod
    def _serialize(array):
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise TypeError("Object dtype is not supported")
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    def write_tensor(self, tensor, **metadata):
        import torch
        with self._lock:
            if not self._has_capacity():
                return None
            event = self._base_event(metadata)
            payloads = {}
            try:
                if not isinstance(tensor, torch.Tensor):
                    event.update(state="unsupported", reason="value is not a tensor")
                    return self._append(event)
                event.update(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device))
                te_fp8 = (type(tensor).__module__, type(tensor).__name__) in _TE_FP8_TYPES
                if not te_fp8 and (tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta"):
                    event.update(state="unsupported", reason="sparse, quantized and meta tensors are not supported")
                    return self._append(event)
                if not te_fp8 and tensor.is_complex():
                    event.update(state="unsupported", reason="complex snapshots are not supported")
                    return self._append(event)
                raw_fp8 = te_fp8 or str(tensor.dtype).startswith("torch.float8")
                if raw_fp8 and (not getattr(self.config, "capture_fp8", False) or self.fp8_adapter is None):
                    event.update(state="unsupported", reason="raw FP8 requires capture_fp8 and an explicit validated adapter")
                    return self._append(event)
                tensor = self._slice(tensor, event)
                event.update(shape=list(tensor.shape), stride=[] if te_fp8 else list(tensor.stride()))
                if self._d2h is not None:
                    if raw_fp8:
                        raise NotImplementedError("async_d2h does not support raw FP8 adapters; use the synchronous snapshot path")
                    return self._queue_d2h(tensor, event)
                arrays = {}
                if raw_fp8:
                    result = self.fp8_adapter(tensor)
                    if not isinstance(result, dict) or not {"payload", "metadata"} <= set(result):
                        raise ValueError("FP8 adapter requires payload and metadata")
                    fp8 = result["metadata"]
                    if not isinstance(fp8, dict) or not {"format", "logical_shape", "scale_mode"} <= set(fp8):
                        raise ValueError("FP8 adapter lacks interpretation metadata")
                    json.dumps(fp8, allow_nan=False)
                    if list(fp8["logical_shape"]) != list(tensor.shape):
                        raise ValueError("FP8 logical shape does not match snapshot")
                    array = np.asarray(result["payload"])
                    if array.dtype != np.uint8 or array.size != tensor.numel():
                        raise ValueError("FP8 payload must contain one uint8 per logical element")
                    event["fp8"] = fp8
                    analysis = result.get("analysis_tensor")
                    if analysis is not None and (not isinstance(analysis, torch.Tensor)
                                                 or list(analysis.shape) != list(tensor.shape)
                                                 or not analysis.is_floating_point()
                                                 or str(analysis.dtype).startswith("torch.float8")):
                        raise ValueError("FP8 analysis_tensor must be a decoded floating tensor of the logical shape")
                    event["stats"] = self._stats(analysis) if analysis is not None else {}
                    for name, auxiliary in result.get("auxiliary", {}).items():
                        if not isinstance(name, str) or not name or not all(c.isalnum() or c == "_" for c in name):
                            raise ValueError("unsafe FP8 auxiliary name")
                        arrays[name] = np.asarray(auxiliary)
                else:
                    event["stats"] = self._stats(tensor)
                    array = None
                event["state"] = "statistics"
                if self.config.mode != "tensor":
                    return self._append(event)
                itemsize = 1 if raw_fp8 else (4 if tensor.dtype == torch.bfloat16 else tensor.element_size())
                predicted = tensor.numel() * itemsize + sum(a.nbytes for a in arrays.values())
                if predicted > self.config.max_tensor_bytes:
                    event.update(state="truncated", reason="max_tensor_bytes reached")
                    return self._append(event)
                if self._reserved_bytes + predicted > self.config.max_rank_bytes:
                    event.update(state="truncated", reason="max_rank_bytes reached")
                    return self._append(event)
                if array is None:
                    value = tensor.detach()
                    if value.dtype == torch.bfloat16:
                        value = value.float()
                    array = value.to("cpu").contiguous().numpy()
                relative = f"tensors/tensor_{event['event_id']:09d}.npy"
                payloads[relative] = self._serialize(array)
                auxiliaries = {}
                for name, value in arrays.items():
                    location = f"tensors/tensor_{event['event_id']:09d}_{name}.npy"
                    payloads[location] = self._serialize(value)
                    auxiliaries[name] = location
                actual = sum(len(value) for value in payloads.values())
                if self._reserved_bytes + actual > self.config.max_rank_bytes:
                    event.update(state="truncated", reason="max_rank_bytes reached")
                    return self._append(event)
                self._reserved_bytes += actual
                event.update(state="tensor", file=relative, stored_dtype=str(array.dtype), bytes=actual)
                if auxiliaries:
                    event["auxiliary"] = auxiliaries
            except AsyncD2HSubmissionError:
                # Ownership may already have reached the worker. Never append
                # a second event with the same event_id after a failed handoff.
                raise
            except NotImplementedError as exc:
                event.update(state="unsupported", reason=str(exc), file=None)
                payloads = {}
            except Exception as exc:
                event.update(state="error", reason=f"{type(exc).__name__}: {exc}", file=None)
                payloads = {}
            return self._append(event, payloads)

    def close(self, status="complete"):
        with self._lock:
            if self._closed:
                return
            self.flush()
            if self._queue is not None:
                self._queue.put(None)
                self._worker.join()
            if self._d2h is not None:
                self._d2h.close()
            try:
                self._stream.close()
            except Exception as exc:
                self._io_errors.append(f"event close failed: {type(exc).__name__}: {exc}")
            if self._io_errors:
                status = "error"
                self.manifest["metadata"]["storage_errors"] = list(self._io_errors)
            if self._counts.get("error", 0):
                status = "error"
            self.manifest.update(status=status, completed_at=_now(), events_count=self.events_count,
                                 tensor_bytes=self.bytes_written, states=dict(self._counts),
                                 event_budget_exhausted=self._event_limit_recorded)
            self._closed = True
            atomic_json(self.rank_dir / "manifest.json", self.manifest)
