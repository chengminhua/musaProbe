"""Opt-in, bounded MUSA snapshots with explicit stream/event ownership.

Only the submitting thread clones live tensors. Worker threads wait on completed
copies and inspect CPU data. Slots are returned after persistence, not after DMA.
"""
from __future__ import annotations

import queue


class AsyncD2HSubmissionError(RuntimeError):
    """Queue handoff failed; ownership is retained until an explicit drain."""


class MusaTransferJob:
    def __init__(self, pool, slot, snapshot, ready, done, device, producer):
        self.pool, self.slot = pool, slot
        self.snapshot, self.ready, self.done = snapshot, ready, done
        self.device, self.producer = device, producer
        self.completed = False
        self.released = False

    def wait(self):
        import torch
        with torch.musa.device(self.device):
            self.done.synchronize()
        self.completed = True
        return self.slot["view"]

    def release(self):
        if self.released:
            return
        self.released = True
        if not self.completed:
            # A failed synchronization cannot prove that DMA released these
            # allocations. Poison the pool and retain them, never reuse them.
            self.pool.poison("MUSA snapshot synchronization did not complete")
            self.pool._quarantine.append(self)
            return
        self.snapshot = self.ready = self.done = None
        self.slot.pop("view", None)
        self.pool._available.put(self.slot)


class MusaTransferPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self._available = queue.Queue(maxsize=capacity)
        for _ in range(capacity):
            self._available.put({"buffer": None})
        self._streams = {}
        self._quarantine = []
        self._abandoned = []
        self._failure = None
        self._closed = False

    def poison(self, reason):
        if self._failure is None:
            self._failure = reason

    def abandon(self, job, reason):
        # queue.put can be interrupted after enqueue but before return. Do not
        # release a possibly worker-owned slot; retain a second strong reference.
        self.poison(reason)
        self._abandoned.append(job)

    def drain_abandoned(self):
        """Called after the disk queue drains, so queued jobs have finished."""
        failures = []
        pending, self._abandoned = self._abandoned, []
        for job in pending:
            if job.released:
                continue
            try:
                job.wait()
            except Exception as error:
                failures.append(f"abandoned D2H drain failed: {type(error).__name__}: {error}")
            finally:
                job.release()
        return failures

    def capture(self, tensor):
        import torch
        if self._closed:
            raise RuntimeError("async_d2h transfer pool is closed")
        # RunWriter serializes submissions with its lock. Autograd invokes
        # accelerator gradient hooks on engine threads; clone inline on that
        # callback's current producer stream before returning to the engine.
        # The disk worker never receives a mutable live training tensor.
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.device.type != "musa" or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.is_complex()
                or str(tensor.dtype).startswith("torch.float8")):
            raise NotImplementedError("async_d2h requires a plain dense MUSA tensor; CPU, FP8, sparse, quantized, complex and tensor subclasses are unsupported")
        api = getattr(torch, "musa", None)
        if api is None or any(not hasattr(api, name) for name in ("Stream", "Event", "stream", "current_stream", "device")):
            raise NotImplementedError("async_d2h requires validated MUSA Stream/Event/pinned-memory APIs")
        while True:
            if self._failure is not None:
                raise RuntimeError(f"async_d2h transfer pool is failed: {self._failure}")
            try:
                slot = self._available.get(timeout=.1)
                break
            except queue.Empty:
                continue
        producer = transfer = snapshot = None
        try:
            index = tensor.device.index
            producer = api.current_stream(index)
            transfer = self._streams.get(index)
            if transfer is None:
                transfer = self._streams[index] = api.Stream(device=index)
            count = tensor.numel()
            buffer = slot["buffer"]
            if buffer is None or buffer.dtype != tensor.dtype or buffer.numel() < max(1, count):
                buffer = torch.empty(max(1, count), dtype=tensor.dtype).pin_memory(device="musa")
                if not buffer.is_pinned(device="musa"):
                    raise NotImplementedError("MUSA pinned host allocation was not pinned; refusing a synchronous fallback")
                slot["buffer"] = buffer
            slot["view"] = buffer[:count].view(tensor.shape)
            # The producer is the current stream at the capture boundary.
            # Subsequent in-place writes on that stream follow this fixed clone.
            with api.stream(producer), torch.no_grad():
                snapshot = tensor.detach().clone(memory_format=torch.contiguous_format)
                ready = api.Event()
                ready.record(producer)
            with api.stream(transfer):
                transfer.wait_event(ready)
                slot["view"].copy_(snapshot, non_blocking=True)
                done = api.Event()
                done.record(transfer)
            return MusaTransferJob(self, slot, snapshot, ready, done, index, producer)
        except BaseException:
            # Failed submission still may have enqueued work. Keep allocations
            # alive until both streams are known to have drained.
            drained = True
            for stream in (producer, transfer):
                if stream is not None:
                    try:
                        stream.synchronize()
                    except Exception as error:
                        drained = False
                        self.poison(f"submission drain failed: {type(error).__name__}: {error}")
            if drained:
                slot.pop("view", None)
                self._available.put(slot)
            else:
                self._quarantine.append((slot, snapshot))
            raise

    def close(self):
        """Drop idle cache only; uncertain failed transfers remain quarantined."""
        self._closed = True
        idle = []
        while True:
            try:
                slot = self._available.get_nowait()
            except queue.Empty:
                break
            slot.clear()
            slot["buffer"] = None
            idle.append(slot)
        for slot in idle:
            self._available.put(slot)
        self._streams.clear()
