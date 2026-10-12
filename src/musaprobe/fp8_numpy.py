"""Bounded NumPy decoding of explicitly described native/TE rowwise FP8."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np


FORMATS = {"float8_e4m3fn": (4, 3, 7), "float8_e5m2": (5, 2, 15),
           "float8_e4m3fnuz": (4, 3, 8), "float8_e5m2fnuz": (5, 2, 16)}


def decode_bits(raw, format_name):
    name = format_name.removeprefix("torch.")
    if name not in FORMATS:
        raise TypeError("unsupported FP8 bit encoding")
    if raw.dtype != np.dtype("uint8"):
        raise ValueError("FP8 bit payload must be uint8")
    exponent_bits, mantissa_bits, bias = FORMATS[name]
    bits = np.arange(256, dtype=np.uint16)
    exponent = ((bits >> mantissa_bits) & ((1 << exponent_bits) - 1)).astype(np.int32)
    mantissa = (bits & ((1 << mantissa_bits) - 1)).astype(np.float32)
    sign = np.where(bits & 128, np.float32(-1), np.float32(1))
    fractions = mantissa / np.float32(1 << mantissa_bits)
    table = np.where(exponent == 0, np.ldexp(fractions, 1 - bias), np.ldexp(1 + fractions, exponent - bias)).astype(np.float32) * sign
    if name.endswith("fnuz"):
        table[128] = np.nan
    elif name == "float8_e4m3fn":
        table[(exponent == 15) & (mantissa == 7)] = np.nan
    else:
        table[(exponent == 31) & (mantissa == 0)] = np.inf * sign[(exponent == 31) & (mantissa == 0)]
        table[(exponent == 31) & (mantissa != 0)] = np.nan
    return table[raw]


def _safe_array(root, reference):
    if not isinstance(reference, str) or not reference:
        raise ValueError("FP8 payload requires a relative file reference")
    relative = Path(reference)
    target = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root.resolve()) or target.suffix != ".npy":
        raise ValueError("unsafe FP8 tensor file reference")
    return np.load(target, mmap_mode="r", allow_pickle=False)


class FP8Tensor:
    """Metric-compatible chunk reader; expansion is at most one metric chunk."""
    def __init__(self, rank_dir, event):
        info = event.get("fp8")
        if not isinstance(info, dict) or not isinstance(info.get("format"), str) or info["format"].removeprefix("torch.") not in FORMATS:
            raise TypeError("unsupported FP8 metadata")
        if info.get("encoding") != "raw_float8_bits":
            raise TypeError("FP8 requires explicit raw_float8_bits encoding")
        shape = info.get("logical_shape")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape) or shape != event.get("shape"):
            raise ValueError("FP8 logical and stored shape metadata disagree")
        self.shape, self.size, self.dtype = tuple(shape), math.prod(shape), np.dtype("float32")
        self.raw = _safe_array(Path(rank_dir), event.get("file"))
        if self.raw.dtype != np.dtype("uint8") or list(self.raw.shape) != shape:
            raise ValueError("FP8 payload shape or dtype differs from metadata")
        numel = event.get("stats", {}).get("numel")
        if numel is not None and numel != self.size:
            raise ValueError("FP8 element count differs from metadata")
        self.info, self.mode, self.scale = info, info.get("scale_mode"), None
        if self.mode not in ("unscaled_native_dtype", "per_tensor", "mt_blockwise"):
            raise TypeError("unsupported FP8 scale mode")
        if self.mode != "unscaled_native_dtype":
            refs = event.get("auxiliary")
            if not isinstance(refs, dict) or "scale_inv" not in refs:
                raise ValueError("scaled FP8 payload requires scale_inv")
            self.scale = _safe_array(Path(rank_dir), refs["scale_inv"])
            if self.scale.dtype != np.dtype("float32"):
                raise ValueError("FP8 scale_inv must be float32")
            for start in range(0, self.scale.size, 262144):
                values = self.scale.flat[start:start + 262144]
                if not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError("FP8 inverse scales must be finite and positive")
            if self.mode == "per_tensor" and self.scale.size != 1:
                raise ValueError("per-tensor inverse scale must contain exactly one element")
            if self.mode == "mt_blockwise":
                bm, bn = info.get("block_m"), info.get("block_n")
                if type(bm) is not int or type(bn) is not int or bm < 1 or bn < 1 or not shape:
                    raise ValueError("invalid MT blockwise scale dimensions")
                cols = shape[-1]
                rows = self.size // cols if cols else 0
                if list(self.scale.shape) != [(rows + bm - 1) // bm, (cols + bn - 1) // bn]:
                    raise ValueError("unsupported MT blockwise scale layout or padding")
        self.flat = self

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("FP8 reader supports only contiguous flat slices")
        start, stop, _ = selection.indices(self.size)
        values = decode_bits(self.raw.flat[start:stop], self.info["format"])
        with np.errstate(over="ignore", invalid="ignore"):
            if self.mode == "per_tensor":
                values = values * self.scale.flat[0]
            elif self.mode == "mt_blockwise" and stop > start:
                indices = np.arange(start, stop)
                cols = self.shape[-1]
                values = values * self.scale[(indices // cols) // self.info["block_m"], (indices % cols) // self.info["block_n"]]
        return values
