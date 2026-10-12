"""Explicit control-plane coordination at quiescent training step boundaries.

The caller must enter on every rank in the same order. A dedicated Gloo group
keeps configuration/stop messages separate from accelerator collectives. A lost
rank poisons this coordinator; no retry or device reset is attempted.
"""
from __future__ import annotations

import datetime
import hashlib
import json


class BoundaryCoordinationError(RuntimeError):
    """A boundary could not be agreed; the caller must stop its training loop."""


def agree_proposals(proposals):
    """Pure consensus validation, also used to test failure paths without torch."""
    if not proposals:
        raise ValueError("No boundary participants")
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Boundary proposals must be dictionaries")
        if type(proposal.get("step")) is not int or proposal["step"] < 0:
            raise ValueError("Boundary step must be a nonnegative integer")
        if type(proposal.get("config_version")) is not int or proposal["config_version"] < 0:
            raise ValueError("Boundary config_version must be a nonnegative integer")
        if type(proposal.get("stop", False)) is not bool:
            raise ValueError("Boundary stop must be boolean")
    first = proposals[0]
    result = {"step": first["step"], "config_version": first["config_version"],
              "stop": any(p.get("stop", False) for p in proposals),
              "update": None, "accepted": False, "error": None}
    if any(p["step"] != first["step"] or p["config_version"] != first["config_version"] for p in proposals):
        result["error"] = "Ranks disagree on boundary step or current configuration version"
        result["stop"] = True
        return result
    errors = [f"rank {rank}: {p['error']}" for rank, p in enumerate(proposals) if p.get("error")]
    if errors:
        result["error"] = "; ".join(errors)
        return result
    # rank/world_size are writer identities, not configurable capture policy.
    # They may legitimately differ between otherwise identical rank configs.
    def policy(update):
        if update is None:
            return None
        if not isinstance(update, dict):
            raise ValueError("Boundary update must be a dictionary or None")
        normalized = {k: v for k, v in update.items() if k not in ("rank", "world_size")}
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(payload.encode()) > 1024 * 1024:
            raise ValueError("Boundary configuration exceeds 1 MiB")
        return hashlib.sha256(payload.encode()).hexdigest()
    fingerprints = [policy(p.get("update")) for p in proposals]
    if len(set(fingerprints)) != 1:
        result["error"] = "Ranks proposed different configurations; previous configuration retained"
        return result
    result.update(accepted=True, update=first.get("update"), config_sha256=fingerprints[0])
    return result


class SafeBoundaryCoordinator:
    """Collective callable for ``PrecisionDebugger.step_boundary(coordinate=...)``.

    Construct and close on all ranks. The existing default group is never
    destroyed. The 30-second default bounds missing-rank failures.
    """

    def __init__(self, *, timeout_seconds=30, group=None):
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be in (0, 600]")
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed on all ranks first")
        self._dist = dist
        self._owns_group = group is None
        self.group = group if group is not None else dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=timeout_seconds))
        if str(dist.get_backend(self.group)).lower() != "gloo":
            raise ValueError("Boundary coordination requires a dedicated Gloo control group")
        self._failed = False
        self._closed = False
        self.history = []

    def __call__(self, proposal):
        if self._closed or self._failed:
            raise BoundaryCoordinationError("Coordinator closed or previously failed")
        # Always participate even if this rank has a bad local proposal, so peers
        # receive a rejection instead of waiting for a rank that exited early.
        local = dict(proposal)
        try:
            agree_proposals([local])
        except (ValueError, TypeError) as error:
            local = {"step": local.get("step", 0), "config_version": local.get("config_version", 0),
                     "update": None, "stop": True, "error": str(error)}
        gathered = [None] * self._dist.get_world_size(self.group)
        try:
            self._dist.all_gather_object(gathered, local, group=self.group)
            result = agree_proposals(gathered)
        except Exception as error:
            self._failed = True
            raise BoundaryCoordinationError(f"Boundary agreement failed; stop training: {error}") from error
        if result["accepted"] and local.get("update") is not None:
            result["update"] = local["update"]
        self.history.append({k: v for k, v in result.items() if k != "update"})
        del self.history[:-100]
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_group:
                self._dist.destroy_process_group(self.group)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
