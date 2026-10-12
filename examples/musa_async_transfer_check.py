#!/usr/bin/env python3
"""Bounded MUSA stream/event D2H correctness experiment, separate from RunWriter.

No training or background capture is launched. Passing demonstrates snapshot
lifetime and ordering for this tensor workload, not transfer/compute overlap.
"""
from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def exercise(*, device_index=0, dtype="float32", iterations=8, elements=4096, buffers=2):
    import torch
    import torch_musa

    result = {"status": "running", "experiment": "musa_stream_event_d2h",
              "production_capture_enabled": False, "dtype": dtype,
              "iterations": iterations, "elements": elements, "buffers": buffers,
              "torch_version": torch.__version__, "torch_musa_version": getattr(torch_musa, "__version__", "unknown"),
              "scope": "immutable_snapshot_ordering_buffer_reuse_and_drain; no_overlap_benchmark",
              "checks": {}}
    checks = result["checks"]
    slots, streams = [], []
    main_thread = threading.get_ident()

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    def consume(slot):
        if slot.get("done") is None:
            return
        slot["done"].synchronize()
        # Retain the device allocation until DMA completes. No record_stream
        # dependency on allocator-specific implementations is needed here.
        expected = torch.full(slot["cpu"].shape, slot["expected"], dtype=slot["cpu"].dtype)
        check(f"snapshot_{slot['iteration']}_exact", torch.equal(slot["cpu"], expected))
        slot["cpu"].fill_(-327.)
        slot["snapshot"] = None
        slot["done"] = None
        slot["uses"] += 1

    try:
        if not torch.musa.is_available():
            raise RuntimeError("MUSA device is unavailable")
        torch.musa.set_device(device_index)
        original_stream = torch.musa.current_stream(device_index)
        producer = torch.musa.Stream(device=device_index)
        transfer = torch.musa.Stream(device=device_index)
        streams.extend((producer, transfer))
        check("two_distinct_nondefault_streams", producer != transfer
              and producer != torch.musa.default_stream(device_index)
              and transfer != torch.musa.default_stream(device_index))
        scalar_type = getattr(torch, dtype)
        with torch.musa.stream(producer):
            source = torch.empty((elements // 2, 2), device=f"musa:{device_index}", dtype=scalar_type)
        for _ in range(buffers):
            cpu = torch.empty((2, elements // 2), dtype=scalar_type).pin_memory(device="musa")
            check("pinned_host_buffers", cpu.is_pinned(device="musa"))
            slots.append({"cpu": cpu, "done": None, "snapshot": None, "uses": 0})
        original_buffers = [slot["cpu"].data_ptr() for slot in slots]
        for index in range(iterations):
            check("snapshot_submission_on_main_thread", threading.get_ident() == main_thread)
            slot = slots[index % buffers]
            consume(slot)  # CPU buffer is reused only after its prior DMA finished.
            expected = float(index + 1)
            with torch.musa.stream(producer):
                source.fill_(expected)
                view = source.transpose(0, 1)
                check("noncontiguous_source_view", not view.is_contiguous())
                snapshot = view.detach().clone(memory_format=torch.contiguous_format)
                ready = torch.musa.Event()
                ready.record(producer)
                # Overwrite the live source immediately after the fixed clone.
                # Transfer must read the snapshot, never this mutable storage.
                source.fill_(-expected)
            slot.update(snapshot=snapshot, expected=expected, iteration=index)
            with torch.musa.stream(transfer):
                transfer.wait_event(ready)
                slot["cpu"].copy_(snapshot, non_blocking=True)
                done = torch.musa.Event()
                done.record(transfer)
                slot["done"] = done
            # Keep the ready event alive as well as the snapshot until drain.
            slot["ready"] = ready
        for slot in slots:
            consume(slot)
        producer.synchronize()
        check("original_source_was_modified", torch.equal(source.cpu(), torch.full(source.shape, -float(iterations), dtype=scalar_type)))
        check("host_buffer_storage_reused", original_buffers == [slot["cpu"].data_ptr() for slot in slots]
              and all(slot["uses"] >= 2 for slot in slots))
        check("caller_stream_restored", torch.musa.current_stream(device_index) == original_stream)
        result["status"] = "pass"
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        # Even on assertion/copy failure, finish submitted work before allowing
        # buffers or snapshots to be released. A device hang needs an external
        # process timeout; Python cannot safely cancel an in-flight DMA.
        drained = True
        for stream in streams:
            try:
                stream.synchronize()
            except Exception as error:
                drained = False
                result.setdefault("drain_errors", []).append(f"{type(error).__name__}: {error}")
        checks["streams_drained"] = drained and bool(streams)
        if not checks["streams_drained"]:
            result["status"] = "error"
        for slot in slots:
            slot["snapshot"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Parent directory, usually under /data")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--buffers", type=int, default=2)
    args = parser.parse_args()
    if not (0 <= args.device_index and 1 <= args.buffers <= 4
            and 2 * args.buffers <= args.iterations <= 32
            and 4 <= args.elements <= 65536 and args.elements % 2 == 0):
        parser.error("require device-index>=0, buffers=1..4, 2*buffers<=iterations<=32, even elements=4..65536")
    output = Path(args.output).expanduser().resolve() / datetime.now(timezone.utc).strftime("async_d2h_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = exercise(device_index=args.device_index, dtype=args.dtype,
                          iterations=args.iterations, elements=args.elements, buffers=args.buffers)
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}", "checks": {}}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**result, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
