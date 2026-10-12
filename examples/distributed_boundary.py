"""Run via torchrun --standalone --nproc-per-node=2; all output is caller-selected."""
import argparse
import datetime
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--scenario", choices=("normal", "divergent", "missing"), default="normal")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from musaprobe import PrecisionDebugger
    from musaprobe.distributed import SafeBoundaryCoordinator, BoundaryCoordinationError
    from musaprobe.integrations.communication import observed_all_reduce
    if args.device == "musa":
        import torch_musa
        torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("mccl" if args.device == "musa" else "gloo",
                            timeout=datetime.timedelta(seconds=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    debugger = PrecisionDebugger(config={"output_dir": str(root), "run_id": "capture",
        "rank": rank, "world_size": world, "mode": "tensor", "dependency_graph": True,
        "stop_on_nonfinite": True, "history_size": 8, "max_events": 100, "stack_depth": 0})
    coordinator = SafeBoundaryCoordinator(timeout_seconds=3 if args.scenario == "missing" else 15)
    results = {"rank": rank, "world_size": world, "device": args.device, "scenario": args.scenario}
    try:
        debugger.start()
        if args.scenario == "missing" and rank == world - 1:
            # Deliberately omit one boundary; this rank remains alive to avoid
            # involving the launcher's process failure handling in the test.
            time.sleep(6)
            results["omitted_boundary"] = True
        elif args.scenario == "missing":
            started = time.monotonic()
            try:
                debugger.step_boundary(0, coordinate=coordinator)
                raise AssertionError("Missing rank should fail agreement")
            except BoundaryCoordinationError:
                results.update(timeout_detected=True, elapsed=time.monotonic() - started)
                assert results["elapsed"] < 6
        else:
            value = torch.tensor([rank + 1.0], device=args.device)
            observed_all_reduce(debugger, value, group_id="world", sequence=0)
            assert value.item() == world * (world + 1) / 2
            results["collective_value"] = value.item()
            debugger.request_config_update({"max_events": 200 if args.scenario == "normal" or rank == 0 else 201})
            response = debugger.step_boundary(1, coordinate=coordinator)
            results["update"] = response
            if args.scenario == "normal":
                assert response["accepted"] and debugger.config.max_events == 200
                if rank == world - 1:
                    debugger.save(torch.tensor([float("nan")], device=args.device), "controlled_nan")
                stop = debugger.step_boundary(2, coordinate=coordinator)
                assert stop["stop"]
                results["coordinated_stop"] = True
            else:
                assert not response["accepted"] and debugger.config.max_events == 100
                results["old_config_preserved"] = True
        results["status"] = "pass"
    finally:
        debugger.close()
        coordinator.close()
        (root / f"rank{rank}.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
