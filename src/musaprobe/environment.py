"""Explicit, bounded training configuration and random/input fingerprints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random


def capture_training_config(*, hyperparameters=None, parallelism=None, precision=None,
                            inputs=None, include_rng=True, max_input_bytes=64 * 1024**2):
    """Return metadata to pass to PrecisionDebugger or persist as a JSON document.

    This reads only specified configuration and inputs, never environment secrets.
    Input copies can synchronize a device; call at an intentional safe boundary.
    A digest records state equality, not a restorable RNG checkpoint.
    """
    import numpy as np
    import torch
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be positive")
    def clean(value):
        return json.loads(json.dumps(value, allow_nan=False))
    result = {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "training": clean(hyperparameters), "parallelism": clean(parallelism),
              "precision": clean(precision), "input_state": None, "random_state": None,
              "environment_scope": {
                  "configuration": "caller_supplied_only",
                  "inputs": "explicit_named_tensors_only" if inputs is not None else "not_captured",
                  "rng": "python_numpy_torch_cpu_and_current_initialized_accelerator" if include_rng else "not_captured",
              }}
    for package, key in (("torch_musa", "torch_musa_version"), ("transformer-engine", "transformer_engine_version")):
        try:
            result[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[key] = None
    def tensor_digest(value):
        tensor = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    if inputs is not None:
        if not isinstance(inputs, dict) or any(not isinstance(k, str) for k in inputs):
            raise ValueError("inputs must be a dictionary of explicit names to tensors")
        budget = 0
        result["input_state"] = {}
        for name, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.device.type == "meta":
                raise ValueError("Input fingerprints require concrete dense tensors")
            budget += tensor.numel() * tensor.element_size()
            if budget > max_input_bytes:
                raise ValueError("Input fingerprint budget exceeded")
            result["input_state"][name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}
    if include_rng:
        numpy_state = np.random.get_state()
        numpy_bytes = numpy_state[1].tobytes() + repr((numpy_state[0], *numpy_state[2:])).encode()
        rng = {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
               "numpy": hashlib.sha256(numpy_bytes).hexdigest(),
               "torch_cpu": tensor_digest(torch.get_rng_state())}
        # Do not initialize a device solely to inspect its RNG state.
        for device in ("musa", "cuda"):
            api = getattr(torch, device, None)
            if api is None or not hasattr(api, "is_initialized") or not api.is_initialized():
                rng[device] = None
                continue
            try:
                rng[device] = {"device": api.current_device(), "sha256": tensor_digest(api.get_rng_state())}
            except (RuntimeError, AttributeError) as error:
                rng[device] = {"status": "unknown", "reason": str(error)}
        result["random_state"] = rng
    return result
