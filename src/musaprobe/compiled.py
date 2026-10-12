"""Real compile/graph execution checks at model output boundaries.

This module does not claim to observe fused kernel internals. Snapshots are
taken after execution, never from Python callbacks inside graph capture.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np


def validate_execution(model, args, output_dir, *, device="cpu", mode="compile", backend=None,
                       replays=2, atol=1e-5, rtol=1e-4):
    import torch
    from .precheck import _metric, _array
    if mode not in ("compile", "graph") or device not in ("cpu", "musa", "cuda"):
        raise ValueError("Unknown execution mode or device")
    if type(replays) is not int or not 2 <= replays <= 16:
        raise ValueError("Use 2..16 replays to verify changed-input execution")
    if not isinstance(args, (tuple, list)) or not args or any(type(x) is not torch.Tensor or not x.is_floating_point() for x in args):
        raise ValueError("Boundary validator accepts a nonempty sequence of ordinary floating tensors")
    if sum(x.numel()*x.element_size() for x in args) > 64*1024**2:
        raise ValueError("Input budget is 64 MiB")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "mode": mode, "device": device, "backend": backend,
              "scope": "eval_model_output_boundary", "status": "unsupported", "replays": [],
              "compiled_graphs": 0, "internal_kernel_coverage": False, "directory": str(root)}
    try:
        if device == "musa":
            import torch_musa  # noqa: F401
        if device != "cpu" and not getattr(torch, device).is_available():
            raise RuntimeError(f"No {device} device is available")
        if backend == "torchada":
            if mode != "compile" or device != "musa":
                raise ValueError("torchada is an explicit MUSA Inductor adapter; use mode=compile, device=musa")
            # The installed adapter's public import applies compatibility patches.
            # It is not a torch._dynamo backend registered under 'torchada'.
            import torchada
            if not torchada.is_musa_platform() or torchada.get_backend() is not torch.musa:
                raise RuntimeError("TorchAda did not select the requested MUSA backend")
            result["adapter"] = {"name": "torchada", "version": torchada.get_version(),
                                 "patches_applied": bool(torchada.is_patched()),
                                 "scope": "process_wide_compatibility_patches"}
            if not result["adapter"]["patches_applied"]:
                raise RuntimeError("TorchAda did not apply its compatibility patches")
            result["resolved_backend"] = "inductor"
        else:
            result["resolved_backend"] = backend or "inductor" if mode == "compile" else None
        eager = copy.deepcopy(model).eval().to(device)
        candidate = copy.deepcopy(model).eval().to(device)
        original = tuple(x.detach().to(device).clone() for x in args)
        graph = None
        if mode == "compile":
            if not hasattr(torch, "compile"): raise RuntimeError("torch.compile is unavailable")
            from torch._dynamo.backends.registry import lookup_backend
            actual_backend = lookup_backend(result["resolved_backend"])
            def counted_backend(gm, example_inputs):
                result["compiled_graphs"] += 1
                return actual_backend(gm, example_inputs)
            candidate = torch.compile(candidate, backend=counted_backend, fullgraph=True)
        else:
            if device == "cpu": raise RuntimeError("Device graph requires MUSA or CUDA")
            api = getattr(torch, device)
            graph_type = getattr(api, "MUSAGraph" if device == "musa" else "CUDAGraph", None)
            if graph_type is None or not hasattr(api, "graph"):
                raise RuntimeError("Installed backend has no matching graph capture API")
            static_inputs = tuple(x.clone() for x in original)
            # Warm-up on a separate stream, then establish explicit dependencies.
            stream = api.Stream()
            stream.wait_stream(api.current_stream())
            with api.stream(stream), torch.no_grad():
                for _ in range(3): candidate(*static_inputs)
            api.current_stream().wait_stream(stream)
            graph = graph_type()
            with torch.no_grad(), api.graph(graph):
                static_output = candidate(*static_inputs)
            if not isinstance(static_output, torch.Tensor): raise ValueError("One tensor output is required")
        with torch.no_grad():
            for index in range(replays):
                inputs = tuple(x + index*0.125 for x in original)
                expected = eager(*inputs)
                if graph is not None:
                    for target, source in zip(static_inputs, inputs): target.copy_(source)
                    graph.replay()
                    actual = static_output
                else:
                    actual = candidate(*inputs)
                if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
                    raise ValueError("One tensor output is required")
                # CPU copies happen after replay; changes in input are persisted.
                for j, value in enumerate(inputs): np.save(root/f"input_{index}_{j}.npy", _array(value), allow_pickle=False)
                np.save(root/f"eager_{index}.npy", _array(expected), allow_pickle=False)
                np.save(root/f"candidate_{index}.npy", _array(actual), allow_pickle=False)
                result["replays"].append({"index": index, "input_offset": index*0.125,
                                          **_metric(expected, actual, atol, rtol)})
        if mode == "compile" and result["compiled_graphs"] == 0:
            raise RuntimeError("No graph reached the requested compiler backend")
        result["status"] = "pass" if all(r["status"] == "pass" for r in result["replays"]) else "mismatch"
    except Exception as exc:
        result.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
    (root/"execution.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result
