"""Owned-kernel debug buffers. No arbitrary device-memory inspection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np


_CPP = "std::vector<torch::Tensor> debug_square(torch::Tensor x);"
_MUSA = r'''
#include <torch/extension.h>
#include <musa_runtime.h>
#include <torch_musa/csrc/core/MUSAStream.h>
__global__ void square_debug(const float* x, float* out, float* intermediate, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float square = x[i] * x[i];
    intermediate[i] = square;
    out[i] = square + 1.0f;
  }
}
std::vector<torch::Tensor> debug_square(torch::Tensor x) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "MUSA input required");
  TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous(), "contiguous float32 required");
  TORCH_CHECK(x.numel() > 0 && x.numel() <= 4096, "bounded 1..4096 element input required");
  int current = -1;
  TORCH_CHECK(musaGetDevice(&current) == musaSuccess && current == x.get_device(), "input must use current device");
  auto out = torch::empty_like(x);
  auto debug = torch::empty_like(x);
  square_debug<<<(x.numel()+255)/256,256,0,c10::musa::getCurrentMUSAStream()>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), debug.data_ptr<float>(), x.numel());
  TORCH_CHECK(musaGetLastError() == musaSuccess, "debug kernel launch failed");
  return {out, debug};
}
'''


def run_debug_kernel(output_dir, *, backend="musa", device="musa", elements=257):
    """Execute one self-owned square/add kernel and verify its explicit intermediate."""
    if backend not in ("musa", "triton") or device != "musa":
        raise ValueError("Supported owned kernels are musa/triton on MUSA")
    if type(elements) is not int or not 1 <= elements <= 4096:
        raise ValueError("elements must be 1..4096")
    root = Path(output_dir).expanduser().resolve(); root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "unsupported", "backend": backend, "device": device,
              "scope": "owned_square_add_kernel_explicit_debug_buffer", "closed_library_internal_dump": False,
              "elements": elements, "directory": str(root)}
    try:
        import torch
        import torch_musa  # noqa: F401
        if not torch.musa.is_available(): raise RuntimeError("No MUSA device is available")
        cpu = torch.linspace(-2, 2, elements, dtype=torch.float32)
        x = cpu.to("musa")
        if backend == "musa":
            from torch_musa.utils.musa_extension import load_inline
            build = root/"build"; build.mkdir()
            (root/"debug_kernel.mu").write_text(_MUSA, encoding="utf-8")
            prior = os.environ.get("MAX_JOBS")
            os.environ["MAX_JOBS"] = "2"
            try:
                module = load_inline(name="musaprobe_debug_"+hashlib.sha256(_MUSA.encode()).hexdigest()[:10],
                                     cpp_sources=_CPP, musa_sources=_MUSA, functions=["debug_square"],
                                     build_directory=str(build), with_musa=True, verbose=False)
            finally:
                if prior is None: os.environ.pop("MAX_JOBS", None)
                else: os.environ["MAX_JOBS"] = prior
            out, debug = module.debug_square(x)
        else:
            global tl
            import triton
            import triton.language as tl
            @triton.jit
            def kernel(X, Y, D, N: tl.constexpr, BLOCK: tl.constexpr):
                offsets = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
                value = tl.load(X+offsets, offsets<N, other=0)
                square = value*value
                tl.store(D+offsets, square, offsets<N)
                tl.store(Y+offsets, square+1.0, offsets<N)
            out, debug = torch.empty_like(x), torch.empty_like(x)
            kernel[(triton.cdiv(elements, 256),)](x, out, debug, elements, BLOCK=256)
        actual, middle = out.cpu().numpy(), debug.cpu().numpy()
        expected = cpu.numpy()**2
        np.save(root/"input.npy", cpu.numpy(), allow_pickle=False)
        np.save(root/"intermediate.npy", middle, allow_pickle=False)
        np.save(root/"output.npy", actual, allow_pickle=False)
        result.update(status="pass" if np.allclose(middle, expected, atol=1e-6, rtol=1e-6) and np.allclose(actual, expected+1, atol=1e-6, rtol=1e-6) else "mismatch",
                      intermediate_max_abs=float(np.max(np.abs(middle-expected))),
                      output_max_abs=float(np.max(np.abs(actual-(expected+1)))))
    except Exception as exc:
        result.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
    (root/"kernel.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def inspect_kernel_interfaces(include_dir):
    """Bounded SDK header inventory; text matches alone do not establish a dump ABI.

    Compiler implementation headers are excluded and MUSA/runtime/library headers
    have priority. A full match buffer never stops scanning subsequent headers.
    """
    root = Path(include_dir).expanduser().resolve()
    if not root.is_dir(): raise ValueError("An existing SDK include directory is required")
    compiler_dirs = {"clang", "clang-c", "llvm", "llvm-c", "llvm-adt", ".git"}
    suffixes = {".h", ".hpp", ".cuh", ".muh"}
    candidates=[]; skipped_compiler_dirs=0; skipped_symlinks=0
    discovery_truncated=False; entries_seen=0
    for directory, dirs, files in os.walk(root, followlinks=False):
        keep=[]
        for name in sorted(dirs):
            if name.lower() in compiler_dirs:
                skipped_compiler_dirs += 1
            elif (Path(directory)/name).is_symlink():
                skipped_symlinks += 1
            else: keep.append(name)
        dirs[:] = keep
        for name in sorted(files):
            entries_seen += 1
            if entries_seen > 20000:
                discovery_truncated=True; break
            path=Path(directory)/name
            if path.suffix.lower() not in suffixes: continue
            if path.is_symlink():
                skipped_symlinks += 1; continue
            candidates.append(path)
        if discovery_truncated: break

    def priority(path):
        relative=str(path.relative_to(root)).lower()
        # Actual public runtime/library families precede unrelated bundled headers.
        preferred=any(token in relative for token in
                      ("musa", "mtgpu", "mccl", "mudnn", "mublas", "mufft", "murand", "musparse", "mupti", "muthrust"))
        return (0 if preferred else 1, relative)

    matches=[]; scanned=[]; errors=[]; total_bytes=0; omitted_matches=0
    oversized_files=0; budget_omitted_files=0
    for path in sorted(candidates, key=priority):
        if len(scanned) >= 2000:
            budget_omitted_files += 1; continue
        try:
            size=path.stat().st_size
            if size > 2*1024**2:
                oversized_files += 1; continue
            if total_bytes+size > 64*1024**2:
                budget_omitted_files += 1; continue
            payload=path.read_bytes()
        except OSError as exc:
            errors.append({"file":str(path.relative_to(root)), "error":f"{type(exc).__name__}: {exc}"})
            continue
        total_bytes += len(payload)
        scanned.append({"file":str(path.relative_to(root)), "bytes":len(payload),
                        "sha256":hashlib.sha256(payload).hexdigest()})
        for line_no, line in enumerate(payload.decode(errors="replace").splitlines(), 1):
            if any(token in line.lower() for token in ("dump", "debugoutput", "debug_output")):
                if len(matches) < 100:
                    matches.append({"file": str(path.relative_to(root)), "line": line_no, "declaration": line.strip()[:300]})
                else: omitted_matches += 1
    truncated=discovery_truncated or bool(budget_omitted_files or oversized_files or omitted_matches or errors)
    return {"headers_checked":len(scanned), "scanned_files":len(scanned), "scanned_bytes":total_bytes,
            "scanned_headers":scanned, "candidate_files":len(candidates), "matches":matches,
            "truncated":truncated, "truncation":{"discovery_limit_reached":discovery_truncated,
                "budget_omitted_files":budget_omitted_files, "oversized_files":oversized_files,
                "omitted_matches":omitted_matches, "read_errors":len(errors)},
            "excluded_compiler_directories":skipped_compiler_dirs, "excluded_symlinks":skipped_symlinks,
            "read_errors":errors, "supported_runtime_dump_verified":False,
            "scope":"bounded_public_header_scan",
            "reason":"Only the listed public headers were inspected; absence of matches is not evidence that no dump interface exists. Text does not establish tensor semantics, API availability or closed-kernel visibility."}
