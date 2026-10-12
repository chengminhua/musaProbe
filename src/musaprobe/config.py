"""Validated capture configuration, independent of PyTorch."""
from __future__ import annotations

import dataclasses
import copy
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]


@dataclass
class ProbeConfig:
    output_dir: str
    run_id: str | None = None
    mode: str = "statistics"
    steps: list[int] | None = None
    ranks: list[int] | None = None
    microbatches: list[int] | None = None
    modules: list[str] = field(default_factory=lambda: ["*"])
    exclude_modules: list[str] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_gradients: bool = True
    capture_parameters: bool = False
    stack_depth: int = 8
    max_tensor_bytes: int = 64 * 1024 * 1024
    max_rank_bytes: int = 8 * 1024 * 1024 * 1024
    max_events: int = 100000
    rank: int | None = None
    world_size: int | None = None
    tensor_slices: list[dict] = field(default_factory=list)
    monitor_every_n_steps: int | None = None
    monitor_start_step: int = 0
    monitor_max_steps: int = 100
    history_size: int = 0
    stop_on_nonfinite: bool = False
    async_write: bool = False
    async_d2h: bool = False
    async_queue_size: int = 4
    high_level_apis: list[str] = field(default_factory=list)
    dependency_graph: bool = False
    capture_fp8: bool = False
    _run_id_explicit: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, os.PathLike):
            self.output_dir = os.fspath(self.output_dir)
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")
        self._run_id_explicit = self.run_id is not None
        if self.run_id is None:
            from_env = os.environ.get("MUSAPROBE_RUN_ID")
            self.run_id = from_env if from_env is not None else _new_run_id()
            self._run_id_explicit = from_env is not None
        if (not isinstance(self.run_id, str) or not self.run_id or
                self.run_id in {".", ".."} or "/" in self.run_id or "\\" in self.run_id or
                any(ord(char) < 32 for char in self.run_id)):
            raise ValueError("run_id must be a nonempty single directory name")
        if self.mode not in ("statistics", "tensor"):
            raise ValueError("mode must be 'statistics' or 'tensor'")
        for name in ("steps", "ranks", "microbatches"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
                    raise ValueError(f"{name} must be a list of nonnegative integers or null")
                setattr(self, name, list(dict.fromkeys(value)))
        for name in ("modules", "exclude_modules", "apis", "high_level_apis"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} must be a list of nonempty glob strings")
            setattr(self, name, list(value))
        for name in ("capture_inputs", "capture_outputs", "capture_gradients", "capture_parameters",
                     "stop_on_nonfinite", "async_write", "async_d2h", "dependency_graph", "capture_fp8"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        if self.async_d2h and not self.async_write:
            raise ValueError("async_d2h requires async_write=true for bounded event/DMA ownership")
        for name in ("stack_depth", "max_tensor_bytes", "max_rank_bytes", "max_events"):
            value = getattr(self, name)
            minimum = 1 if name == "max_events" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("rank", "world_size"):
            value = getattr(self, name)
            minimum = 1 if name == "world_size" else 0
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum} or null")
        if self.rank is not None and self.world_size is not None and self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.world_size is not None and self.world_size > 1 and not self._run_id_explicit:
            raise ValueError("distributed capture requires an explicit shared run_id or MUSAPROBE_RUN_ID")
        for name, minimum in (("monitor_start_step", 0), ("monitor_max_steps", 1),
                              ("history_size", 0), ("async_queue_size", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.monitor_every_n_steps is not None and (type(self.monitor_every_n_steps) is not int or self.monitor_every_n_steps < 1):
            raise ValueError("monitor_every_n_steps must be a positive integer or null")
        if not isinstance(self.tensor_slices, list):
            raise ValueError("tensor_slices must be a list")
        self.tensor_slices = copy.deepcopy(self.tensor_slices)
        for rule in self.tensor_slices:
            if (not isinstance(rule, dict) or set(rule) != {"module", "kind", "tensor_path", "selection"}
                    or any(not isinstance(rule[k], str) or not rule[k] for k in ("module", "kind", "tensor_path"))
                    or not isinstance(rule["selection"], list)):
                raise ValueError("slice rules require module/kind/tensor_path globs and selection")
            for axis in rule["selection"]:
                if (not isinstance(axis, list) or len(axis) != 3
                        or any(v is not None and type(v) is not int for v in axis)
                        or (axis[2] is not None and axis[2] <= 0)):
                    raise ValueError("slice axes must be [start,stop,positive_step], with optional nulls")

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ProbeConfig":
        if not isinstance(mapping, Mapping):
            raise ValueError("configuration must be an object")
        allowed = {item.name for item in dataclasses.fields(cls) if item.init}
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError("unknown configuration fields: " + ", ".join(sorted(map(str, unknown))))
        if "output_dir" not in mapping:
            raise ValueError("output_dir is required")
        return cls(**dict(mapping))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ProbeConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in dataclasses.asdict(self).items() if not key.startswith("_")}
