"""Verify safe-boundary failure after one owned rank process exits abruptly.

The parent owns both child processes, checks the intentional exit code, and
terminates only these children if the 45-second deadline is exceeded.
"""
import argparse
import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time


def worker(rank, root, device):
    import torch
    import torch.distributed as dist
    from musaprobe import PrecisionDebugger
    from musaprobe.distributed import SafeBoundaryCoordinator, BoundaryCoordinationError
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device == "musa":
        import torch_musa
        torch.musa.set_device(rank)
        assert torch.ones(1, device=f"musa:{rank}").item() == 1
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=10))
    coordinator = SafeBoundaryCoordinator(timeout_seconds=3)
    dist.barrier()
    if rank == 1:
        (root / "rank1.json").write_text(json.dumps({"intentional_exit": 7}) + "\n")
        os._exit(7)
    probe = PrecisionDebugger(config={"output_dir": str(root), "run_id": "capture", "rank": 0, "world_size": 2})
    result = {"status": "error", "device": device, "scope": "dedicated_gloo_control_plane_after_rank_exit"}
    started = time.monotonic()
    try:
        probe.start()
        try:
            probe.step_boundary(0, coordinate=coordinator)
            raise AssertionError("A lost participant must fail agreement")
        except BoundaryCoordinationError as error:
            result.update(reason=str(error), elapsed=time.monotonic() - started)
        assert result["elapsed"] < 8
        started = time.monotonic()
        try:
            probe.step_boundary(1, coordinate=coordinator)
            raise AssertionError("A failed coordinator must not be reused")
        except BoundaryCoordinationError:
            assert time.monotonic() - started < 1
        result.update(status="pass", subsequent_collective_rejected=True)
    finally:
        probe.close()
        coordinator.close()
        dist.destroy_process_group()
        (root / "rank0.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    ctx = multiprocessing.get_context("spawn")
    children = [ctx.Process(target=worker, args=(rank, root, args.device)) for rank in range(2)]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 45
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
        codes = [child.exitcode for child in children]
        assert codes == [0, 7], codes
        result = json.loads((root / "rank0.json").read_text())
        assert result["status"] == "pass", result
        result["child_exit_codes"] = codes
        (root / "rank_loss.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
            if child.is_alive():
                child.kill()
                child.join(5)


if __name__ == "__main__":
    main()
