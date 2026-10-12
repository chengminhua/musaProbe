"""Bounded forward/VJP checks, distinguishing synthetic from captured inputs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .replay import PURE_OPS, load_captured_call


def _array(tensor):
    return tensor.detach().to(device="cpu", dtype=__import__("torch").float64).numpy()


def _metric(expected, actual, atol, rtol):
    from .compare import _metrics
    result = _metrics(_array(expected), _array(actual), atol, rtol)
    result["status"] = "pass" if not (result["mismatch_count"] or result["nonfinite_reference"] or result["nonfinite_candidate"]) else "mismatch"
    return result


def _synthetic(op, dtype, seed):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64).to(dtype)
    x = rand(2, 4)
    if op in ("aten.mm.default", "aten.matmul.default"):
        return (x, rand(4, 3)), {}
    if op == "aten.bmm.default":
        return (rand(2, 3, 4), rand(2, 4, 3)), {}
    if op == "aten.linear.default":
        return (x, rand(3, 4), rand(3)), {}
    if op == "aten.layer_norm.default":
        return (x, [4], rand(4), rand(4), 1e-5, False), {}
    if op == "aten.softmax.int":
        return (x, -1), {}
    if op == "aten.gelu.default":
        return (x,), {"approximate": "none"}
    if op == "aten.pow.Tensor_Scalar":
        return (x, 2.0), {}
    if op in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"):
        y = rand(2, 4)
        if op == "aten.div.Tensor":
            y = y.abs() + 0.5
        return (x, y), {}
    return (x,), {}


def _check(op, args, kwargs, output_dir, *, device, backward, gradient_seed, atol, rtol, provenance):
    import torch
    if op not in PURE_OPS:
        raise ValueError("Only PURE_OPS can be prechecked")
    if device not in ("cpu", "musa", "cuda"):
        raise ValueError("device must be cpu, musa or cuda")
    if any(not np.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "op": op, "device": device, "reference_device": "cpu",
              "reference_dtype": "float64", "status": "error", "provenance": provenance,
              "backward_scope": "synthetic_vector_jacobian_product" if backward else "not_requested",
              "gradient_seed": gradient_seed, "atol": atol, "rtol": rtol,
              "directory": str(directory), "checks": [], "files": {}}
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA device is available")
        if device == "musa":
            import torch_musa  # noqa: F401
            if not torch.musa.is_available():
                raise RuntimeError("No MUSA device is available")
        originals = []
        seen = {}
        def encode(v):
            if isinstance(v, torch.Tensor):
                if v.layout != torch.strided or not v.is_floating_point() or str(v.dtype).startswith("torch.float8"):
                    raise ValueError("VJP precheck currently requires dense non-FP8 floating inputs")
                if id(v) not in seen:
                    seen[id(v)] = len(originals)
                    originals.append(v.detach().cpu())
                return {"tensor_index": seen[id(v)]}
            if isinstance(v, tuple): return {"tuple": [encode(x) for x in v]}
            if isinstance(v, list): return {"list": [encode(x) for x in v]}
            if isinstance(v, dict): return {"dict": {k: encode(x) for k, x in v.items()}}
            if v is None or type(v) in (bool, int, float, str): return v
            raise ValueError("Unsupported argument object")
        spec_args, spec_kwargs = encode(args), encode(kwargs)
        if sum(x.numel() * 8 for x in originals) > 64 * 1024**2:
            raise ValueError("Precheck input budget is 64 MiB at reference precision")
        def save(name, tensor):
            filename = name + ".npy"
            np.save(directory / filename, _array(tensor), allow_pickle=False)
            result["files"][name] = filename
        for i, value in enumerate(originals): save(f"input_{i}", value)
        result["input_dtypes"] = [str(x.dtype) for x in originals]
        result["call_spec"] = {"args": spec_args, "kwargs": spec_kwargs}
        def execute(target, reference):
            values = [x.to(device=target, dtype=torch.float64 if reference else x.dtype).clone().requires_grad_(backward) for x in originals]
            def decode(v):
                if not isinstance(v, dict): return v
                if "tensor_index" in v: return values[v["tensor_index"]]
                if "tuple" in v: return tuple(decode(x) for x in v["tuple"])
                if "list" in v: return [decode(x) for x in v["list"]]
                if "dict" in v: return {k: decode(x) for k, x in v["dict"].items()}
                raise ValueError("Invalid internal call tree")
            _, packet, overload = op.split(".")
            out = getattr(getattr(torch.ops.aten, packet), overload)(*decode(spec_args), **decode(spec_kwargs))
            if not isinstance(out, torch.Tensor) or not out.is_floating_point():
                raise ValueError("Precheck requires one floating output")
            return out, values
        # Bound broadcast/matmul expansion before allocating any real output.
        abstract_output, _ = execute("meta", True)
        if abstract_output.numel()*8 > 64*1024**2:
            raise ValueError("Precheck output exceeds the 64 MiB reference budget")
        reference, ref_inputs = execute("cpu", True)
        candidate, cand_inputs = execute(device, False)
        save("reference_output", reference); save("candidate_output", candidate)
        result["checks"].append({"kind": "forward", **_metric(reference, candidate, atol, rtol)})
        if backward:
            generator = torch.Generator(device="cpu").manual_seed(gradient_seed)
            # The same representable seed is used by both VJPs and persisted.
            seed = torch.randn(reference.shape, generator=generator, dtype=torch.float64).to(candidate.dtype).double()
            save("gradient_seed", seed)
            ref_grads = torch.autograd.grad(reference, ref_inputs, grad_outputs=seed, allow_unused=True)
            cand_grads = torch.autograd.grad(candidate, cand_inputs, grad_outputs=seed.to(device=device, dtype=candidate.dtype), allow_unused=True)
            for i, (a, b) in enumerate(zip(ref_grads, cand_grads)):
                if a is None or b is None:
                    result["checks"].append({"kind": "input_grad", "index": i,
                                             "status": "pass" if a is b else "mismatch", "unused": True})
                    continue
                save(f"reference_grad_{i}", a); save(f"candidate_grad_{i}", b)
                result["checks"].append({"kind": "input_grad", "index": i, **_metric(a, b, atol, rtol)})
        result["status"] = "pass" if all(c["status"] == "pass" for c in result["checks"]) else "mismatch"
        result["reason"] = "CPU float64 is a numerical reference, not an unconditional exact oracle; no model/optimizer state is restored."
    except (ImportError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        result.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
    (directory / "precheck.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def run_precheck(op, output_dir, *, device="cpu", dtype="float32", seed=0, backward=True, atol=1e-5, rtol=1e-4):
    """Create one deterministic synthetic case and persist its inputs and VJP seed."""
    import torch
    if op not in PURE_OPS: raise ValueError("Unknown or unsafe op")
    if dtype not in ("float16", "bfloat16", "float32", "float64"): raise ValueError("Unsupported dtype")
    args, kwargs = _synthetic(op, getattr(torch, dtype), seed)
    return _check(op, args, kwargs, output_dir, device=device, backward=backward,
                  gradient_seed=seed+1, atol=atol, rtol=rtol,
                  provenance={"kind": "synthetic", "input_seed": seed})


def precheck_captured(rank_dir, event_id, output_dir, *, device="cpu", backward=True, gradient_seed=0, atol=1e-5, rtol=1e-4):
    """Check captured inputs using a saved synthetic VJP, not original training gradients."""
    op, args, kwargs = load_captured_call(rank_dir, event_id)
    return _check(op, args, kwargs, output_dir, device=device, backward=backward,
                  gradient_seed=gradient_seed, atol=atol, rtol=rtol,
                  provenance={"kind": "captured_inputs", "rank_dir": str(Path(rank_dir).resolve()), "event_id": event_id,
                              "captured_output_checked": False, "original_training_backward_replayed": False})
