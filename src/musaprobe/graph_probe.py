"""Explicit fixed-buffer probe points that execute inside compiled/device graphs.

Insert a GraphTensorProbe at a model location you own. Read snapshots only after
execution/replay. It observes that tensor boundary, not opaque kernel internals.
"""
from __future__ import annotations

import torch


class GraphTensorProbe(torch.nn.Module):
    def __init__(self, shape, *, dtype=torch.float32, device="cpu", max_bytes=64 * 1024**2):
        super().__init__()
        if (not isinstance(shape, (tuple, list)) or any(type(x) is not int or x <= 0 for x in shape)
                or type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("Provide positive static dimensions and a positive byte budget")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Graph probes support ordinary floating-point tensors")
        import math
        if math.prod(shape) * torch.empty((), dtype=dtype).element_size() > max_bytes:
            raise ValueError("Graph probe buffer exceeds max_bytes")
        self.register_buffer("snapshot_buffer", torch.empty(tuple(shape), dtype=dtype, device=device), persistent=False)
        self.register_buffer("generation", torch.zeros((), dtype=torch.int64, device=device), persistent=False)
        self._last_read_generation = 0

    def forward(self, value):
        if value.shape != self.snapshot_buffer.shape or value.dtype != self.snapshot_buffer.dtype or value.device != self.snapshot_buffer.device:
            raise ValueError("Graph probe requires its declared static shape, dtype and device")
        with torch.no_grad():
            self.snapshot_buffer.copy_(value.detach())
            self.generation.add_(1)
        return value

    def snapshot(self, *, require_fresh=True):
        """Synchronize and copy outside replay/capture; detect stale reads."""
        generation = int(self.generation.item())
        if generation == 0 or (require_fresh and generation <= self._last_read_generation):
            raise RuntimeError("Graph probe has no new execution to observe")
        value = self.snapshot_buffer.detach().clone()
        self._last_read_generation = generation
        return value, generation

    def save(self, debugger, name, *, execution_mode):
        if execution_mode not in ("eager", "compile", "musa_graph", "cuda_graph"):
            raise ValueError("Unknown execution mode")
        value, generation = self.snapshot()
        with debugger.phase("forward"):
            debugger.set_context(execution_mode=execution_mode, graph_generation=generation,
                                 graph_probe_scope="explicit_model_tensor_boundary")
            debugger.save(value, name, kind="graph_tensor")
        return generation
