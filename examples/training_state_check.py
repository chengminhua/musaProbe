#!/usr/bin/env python3
"""Small real optimizer, distributed-gradient and MCore checkpoint acceptance.

Each invocation requires a new output directory. ``shards`` uses exactly two
torchrun processes; it partitions a real all-reduced gradient by an explicit
rectangle declaration, without claiming Megatron optimizer ownership. The MCore
mode imports the caller's actual source tree and never substitutes a serializer.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from musaprobe import PrecisionDebugger
from musaprobe.analysis import build_trends
from musaprobe.checkpoint import compare_checkpoints, convert_checkpoint
from musaprobe.integrations.optimizer import capture_optimizer_state
from musaprobe.layout_axes import reconstruct_axes
from musaprobe.storage import atomic_json


class Checks:
    def __init__(self):
        self.values = {}

    def __call__(self, name, condition):
        self.values[name] = bool(condition)
        if not condition:
            raise AssertionError(name)


def _events(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _model(device, dtype=torch.float32):
    # Fixed initialization avoids relying on cross-device RNG equivalence.
    model = torch.nn.Linear(4, 4, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.arange(16, device=device, dtype=dtype).reshape(4, 4) / 32)
        model.bias.copy_(torch.arange(4, device=device, dtype=dtype) / 16)
    return model


def _batch(device, dtype, step=0, rank=0):
    return torch.arange(8, device=device, dtype=dtype).reshape(2, 4) / 8 + step / 10 + rank


def _probe(output, name, *, rank=0, world=1, steps=None):
    return PrecisionDebugger(config={
        "output_dir": str(output / "captures"), "run_id": name, "mode": "tensor",
        "rank": rank, "world_size": world, "steps": steps, "stack_depth": 0,
        "max_events": 100, "max_tensor_bytes": 4096, "max_rank_bytes": 1024 * 1024,
    }).start()


def optimizer_check(output, device, dtype=torch.float32):
    """Observe actual AdamW states after three completed optimizer steps."""
    output = Path(output)
    check = Checks()
    observed = _model(device, dtype)
    baseline = copy.deepcopy(observed)
    opt = torch.optim.AdamW(observed.parameters(), lr=.01, foreach=False)
    control = torch.optim.AdamW(baseline.parameters(), lr=.01, foreach=False)
    expected = {}
    with _probe(output, "optimizer", steps=[0, 1, 2]) as probe:
        for step in range(3):
            losses = []
            for model, optimizer in ((baseline, control), (observed, opt)):
                optimizer.zero_grad(set_to_none=True)
                loss = model(_batch(device, dtype, step)).float().square().mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
            check(f"step_{step}_loss_unchanged", torch.equal(*losses))
            snapshot = {name + "." + key: value.detach().clone()
                        for name, parameter in observed.named_parameters()
                        for key, value in opt.state[parameter].items() if isinstance(value, torch.Tensor)}
            cpu_rng = torch.get_rng_state().clone()
            device_rng = torch.musa.get_rng_state().clone() if device == "musa" else None
            probe.set_context(step=step, microbatch=None)
            result = capture_optimizer_state(probe, observed, opt, stage="post_step")
            check(f"step_{step}_capture_complete", result["status"] == "pass" and len(result["captured"]) == 6)
            check(f"step_{step}_rng_unchanged", torch.equal(cpu_rng, torch.get_rng_state()) and
                  (device_rng is None or torch.equal(device_rng, torch.musa.get_rng_state())))
            check(f"step_{step}_state_unchanged", all(torch.equal(snapshot[name + "." + key], value)
                  for name, parameter in observed.named_parameters() for key, value in opt.state[parameter].items()
                  if isinstance(value, torch.Tensor)))
            check(f"step_{step}_parameters_unchanged", all(torch.equal(a, b)
                  for a, b in zip(observed.parameters(), baseline.parameters())))
            expected.update({(step, key): _numpy(value) for key, value in snapshot.items()})
    events = _events(probe.rank_dir)
    check("all_18_state_payloads_complete", len(events) == 18 and all(e["state"] == "tensor" and e.get("file") for e in events))
    check("saved_states_equal_actual_optimizer", all(np.array_equal(
        np.load(probe.rank_dir / event["file"], allow_pickle=False), expected[(event["step"], event["module"])]) for event in events))
    check("manifest_complete", json.loads((probe.rank_dir / "manifest.json").read_text())["status"] == "complete")
    trends = build_trends(probe.rank_dir.parent, output=output / "trends")
    check("six_complete_three_step_trends", trends["status"] == "complete" and len(trends["series"]) == 6 and
          all([point["step"] for point in series["points"]] == [0, 1, 2] and
              all(point["state"] == "available" for point in series["points"]) for series in trends["series"]))
    return {"status": "passed", "mode": "optimizer", "device": device, "dtype": str(dtype),
            "assertions": check.values, "events": len(events), "trend_series": len(trends["series"]),
            "scope": "Ordinary torch.optim.AdamW actual model-parameter ownership; three tiny steps; no master-weight or distributed-optimizer adapter claim."}


def shards_check(output, device, rank, world):
    """Two actual backward/all-reduce gradients; explicitly declared row shards."""
    if world != 2:
        raise ValueError("shards mode requires torchrun --nproc-per-node=2")
    check = Checks()
    model = _model(device)
    start, stop = rank * 2, (rank + 1) * 2
    with _probe(output, "gradient_rows", rank=rank, world=world, steps=[0]) as probe:
        probe.set_context(step=0, microbatch=None, phase="backward",
                          parameter_name="weight", gradient_stage="post_all_reduce",
                          partition_source="explicit_validation_rows_of_real_parameter_gradient")
        with probe.backward_session("reduced-gradient"):
            model(_batch(device, torch.float32, rank=rank)).square().mean().backward()
            torch.distributed.all_reduce(model.weight.grad)
            model.weight.grad.div_(world)
            full = model.weight.grad.detach().clone()
            probe.save(model.weight.grad[start:stop], "weight", kind="parameter_grad")
    check("capture_did_not_mutate_gradient", torch.equal(full, model.weight.grad))
    events = _events(probe.rank_dir)
    check("rank_has_one_complete_real_gradient", len(events) == 1 and events[0]["state"] == "tensor" and events[0].get("file"))
    atomic_json(output / f"rank_{rank}_checks.json", check.values)
    torch.distributed.barrier()
    if rank:
        return None
    declaration = {"schema_version": 1, "groups": [{"name": "weight", "global_shape": [4, 4], "shards": [
        {"rank": owner, "event_id": 0, "offsets": [owner * 2, 0]} for owner in range(world)]}]}
    atomic_json(output / "gradient_layout.json", declaration)
    result = reconstruct_axes(probe.rank_dir.parent, declaration, output / "reconstructed", max_bytes=4096)
    check("all_ranks_complete", all(all(json.loads((output / f"rank_{owner}_checks.json").read_text()).values()) for owner in range(world)))
    check("explicit_rectangles_complete", result["status"] == "complete")
    restored = np.load(output / "reconstructed/rank_00000/tensor_00000000.npy", allow_pickle=False)
    check("reconstruction_equals_actual_reduced_gradient", np.array_equal(restored, _numpy(full)))
    # Independent CPU autograd calculation checks the distributed reduction.
    reference = _model("cpu")
    sum(reference(_batch("cpu", torch.float32, rank=owner)).square().mean() / world for owner in range(world)).backward()
    check("reduced_gradient_matches_independent_cpu_autograd", np.allclose(restored, _numpy(reference.weight.grad), atol=1e-5, rtol=1e-5))
    bad = copy.deepcopy(declaration)
    bad["groups"][0]["shards"][1]["offsets"] = [0, 0]
    rejected = reconstruct_axes(probe.rank_dir.parent, bad, output / "overlap_rejected", max_bytes=4096)
    check("overlap_is_rejected", rejected["status"] == "incomplete")
    np.save(output / "actual_reduced_gradient.npy", _numpy(full), allow_pickle=False)
    return {"status": "passed", "mode": "shards", "device": device, "world_size": world,
            "assertions": check.values, "collective_backend": str(torch.distributed.get_backend()),
            "scope": "Real parameter backward and all-reduce, explicit two-row validation partitions. This is not a Megatron optimizer-shard ownership or TP/PP/VPP test."}


def _save_mcore_with_cleanup(dist_checkpointing, state, checkpoint, output, rank):
    """Audit the actual target version's temporal writer and Manager lifetime.

    Error cleanup never calls finalizers or collectives on a failed group. It
    only joins writers queued by this isolated save and shuts down the Manager
    created by it. A stalled writer is reported, not killed or counted as saved.
    """
    from megatron.core.dist_checkpointing.strategies import base, filesystem_async
    queue = base.async_calls
    if queue.get_num_unfinalized_calls():
        raise RuntimeError("isolated MCore acceptance requires an empty existing async-save queue")
    previous_proxy = filesystem_async._results_queue
    audit = {"queue_before": 0, "save_returned": False, "joined_writers": [],
             "new_manager_shutdown": False, "cleanup_complete": True}
    try:
        dist_checkpointing.save(state, str(checkpoint), sharded_strategy=("torch_dist", 1),
                               validate_access_integrity=True, async_sharded_save=False)
        audit["save_returned"] = True
    finally:
        audit["queue_after_save"] = queue.get_num_unfinalized_calls()
        # The normal synchronous implementation has already joined and removed
        # these requests. A failed synchronization can leave a writer pending.
        for request in list(queue.async_calls):
            process = getattr(request.async_caller, "process", None)
            if process is None:
                continue
            try:
                process.join(timeout=10)
                alive = process.is_alive()
                audit["joined_writers"].append({"pid": process.pid, "alive": alive, "exitcode": process.exitcode})
                audit["cleanup_complete"] = audit["cleanup_complete"] and not alive
            except Exception as exc:
                audit["cleanup_complete"] = False
                audit.setdefault("cleanup_errors", []).append(type(exc).__name__ + ": " + str(exc))
        proxy = filesystem_async._results_queue
        if proxy is not None and proxy is not previous_proxy and audit["cleanup_complete"]:
            manager = getattr(proxy, "_manager", None)
            try:
                if manager is None:
                    raise RuntimeError("target Manager proxy has no accessible owner for explicit cleanup")
                manager.shutdown()
                filesystem_async._results_queue = None
                audit["new_manager_shutdown"] = True
            except Exception as exc:
                audit["cleanup_complete"] = False
                audit.setdefault("cleanup_errors", []).append(type(exc).__name__ + ": " + str(exc))
        atomic_json(output / f"mcore_cleanup_rank_{rank}.json", audit)
        if audit["save_returned"] and (audit["queue_after_save"] or not audit["cleanup_complete"]):
            raise RuntimeError("MCore save returned without a fully drained writer lifecycle; see cleanup audit")
    return audit


def mcore_checkpoint_check(output, device, rank, world, source, patch_source=None):
    """Serialize with the supplied real MCore save API, then safely read it."""
    source = Path(source).resolve()
    if not (source / "megatron/core/dist_checkpointing/serialization.py").is_file():
        raise ValueError("--mcore-path must contain the actual Megatron source tree")
    sys.path.insert(0, str(source))
    if patch_source is not None:
        if device != "musa":
            raise ValueError("--musa-patch-path is only valid with --device musa")
        patch_source = Path(patch_source).resolve()
        if not (patch_source / "musa_patch/__init__.py").is_file():
            raise ValueError("--musa-patch-path must contain the target musa_patch package")
        sys.path.insert(0, str(patch_source))
        patch = importlib.import_module("musa_patch")
        if not Path(patch.__file__).resolve().is_relative_to(patch_source):
            raise RuntimeError("musa_patch was imported from a different source tree")
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    actual_source = Path(inspect.getfile(dist_checkpointing)).resolve()
    if not actual_source.is_relative_to(source):
        raise RuntimeError("MCore was imported from a different source tree")
    check = Checks()
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    model(_batch(device, torch.float32)).square().mean().backward()
    optimizer.step()
    if 4 % world:
        raise ValueError("world size must divide the four weight rows")
    rows = 4 // world
    weight_before = model.weight.detach().clone()
    shard = model.weight.detach()[rank * rows:(rank + 1) * rows].clone()
    sharded = ShardedTensor.from_rank_offsets("model.weight", shard, (0, rank, world))
    checkpoint = output / "mcore_torch_dist"
    if rank == 0:
        checkpoint.mkdir()
    torch.distributed.barrier()
    cleanup = _save_mcore_with_cleanup(dist_checkpointing, {"model": {"weight": sharded}}, checkpoint, output, rank)
    check("mcore_writer_queue_drained", cleanup["queue_after_save"] == 0 and cleanup["cleanup_complete"])
    check("mcore_save_did_not_mutate_parameter", torch.equal(weight_before, model.weight))
    atomic_json(output / f"rank_{rank}_checks.json", check.values)
    torch.distributed.barrier()
    if rank:
        return None
    check("real_torch_dist_metadata_and_payload", (checkpoint / ".metadata").is_file() and bool(list(checkpoint.glob("*.distcp"))))
    converted = convert_checkpoint(checkpoint, output / "converted", max_bytes=1024 * 1024)
    check("actual_mcore_checkpoint_converts", converted["status"] == "complete")
    torch.save({"model.weight": weight_before.detach().cpu()}, output / "reference.pt")
    reference = convert_checkpoint(output / "reference.pt", output / "reference", max_bytes=1024 * 1024)
    check("independent_actual_weight_reference_converts", reference["status"] == "complete")
    compared = compare_checkpoints(output / "reference", output / "converted", output=output / "comparison")
    check("mcore_checkpoint_matches_actual_weight", compared["overall"] == "pass")
    changed = weight_before.detach().cpu().clone()
    changed[0, 0] += 1
    torch.save({"model.weight": changed}, output / "changed.pt")
    convert_checkpoint(output / "changed.pt", output / "changed", max_bytes=1024 * 1024)
    check("intentional_weight_difference_detected", compare_checkpoints(output / "changed", output / "converted")["overall"] == "fail")
    check("all_rank_parameters_unchanged", all(all(json.loads((output / f"rank_{owner}_checks.json").read_text()).values()) for owner in range(world)))
    paths = ("megatron/core/dist_checkpointing/serialization.py", "megatron/core/dist_checkpointing/mapping.py", "megatron/core/dist_checkpointing/strategies/torch.py")
    return {"status": "passed", "mode": "mcore-checkpoint", "device": device, "world_size": world,
            "assertions": check.values, "source_hashes": {path: hashlib.sha256((source / path).read_bytes()).hexdigest() for path in paths},
            "serializer": "actual megatron.core.dist_checkpointing.save, torch_dist v1",
            "scope": "Tiny regular rectangular ShardedTensor weight after an actual AdamW step. No flattened N-D optimizer checkpoint, resume, TP/PP/VPP or full-model checkpoint claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("optimizer", "shards", "mcore-checkpoint"))
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="optimizer mode only; other modes require float32")
    parser.add_argument("--output", required=True, type=Path, help="new output directory; must not already exist")
    parser.add_argument("--mcore-path", type=Path)
    parser.add_argument("--musa-patch-path", type=Path)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if args.mode == "optimizer" and world != 1:
        parser.error("optimizer mode requires a single process")
    if args.mode == "shards" and world != 2:
        parser.error("shards mode requires torchrun --nproc-per-node=2")
    if args.mode != "optimizer" and args.dtype != "float32":
        parser.error("only optimizer mode supports --dtype bfloat16")
    if args.mode == "mcore-checkpoint" and args.mcore_path is None:
        parser.error("mcore-checkpoint requires --mcore-path")
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    initialized = False
    try:
        if args.device == "musa":
            import torch_musa  # noqa: F401
            torch.musa.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if args.mode in ("shards", "mcore-checkpoint"):
            # The real MCore synchronous save finalizer also all-reduces a
            # device tensor; its default group must support MUSA even at rank 0.
            backend = "mccl" if args.device == "musa" else "gloo"
            options = ({"init_method": "env://"} if world > 1 else
                       {"init_method": "file://" + str(output / "rendezvous"), "rank": 0, "world_size": 1})
            torch.distributed.init_process_group(backend, timeout=timedelta(seconds=90), **options)
            initialized = True
            torch.distributed.barrier()
        if args.mode == "optimizer":
            result = optimizer_check(output, args.device, getattr(torch, args.dtype))
        elif args.mode == "shards":
            result = shards_check(output, args.device, rank, world)
        else:
            result = mcore_checkpoint_check(output, args.device, rank, world, args.mcore_path, args.musa_patch_path)
        if initialized:
            torch.distributed.barrier()
    except BaseException as exc:
        if output.exists():
            atomic_json(output / f"error_rank_{rank}.json", {"status": "failed", "mode": args.mode,
                        "error_type": type(exc).__name__, "reason": str(exc), "rank": rank})
        raise
    finally:
        if initialized:
            torch.distributed.destroy_process_group()
    # A completed summary is only written after process-group teardown returns.
    if rank == 0:
        result["torch_version"] = str(torch.__version__)
        if args.device == "musa":
            result["torch_musa_version"] = str(getattr(torch_musa, "__version__", "unknown"))
        atomic_json(output / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
