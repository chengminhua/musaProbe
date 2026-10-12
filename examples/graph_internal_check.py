"""Validate an explicit tensor probe inside a real compiled graph or device graph."""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("compile", "graph"), required=True)
    parser.add_argument("--device", choices=("cpu", "musa"), default="musa")
    args = parser.parse_args()
    import torch
    if args.device == "musa":
        import torch_musa
    from musaprobe import PrecisionDebugger
    from musaprobe.graph_probe import GraphTensorProbe
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(31)
    linear = torch.nn.Linear(4, 3).eval().to(args.device)
    reference = copy.deepcopy(linear)
    point = GraphTensorProbe((2, 3), device=args.device)
    model = torch.nn.Sequential(linear, point, torch.nn.GELU()).eval()
    initial = torch.arange(8, dtype=torch.float32, device=args.device).reshape(2, 4)
    result = {"mode": args.mode, "device": args.device, "status": "error", "generations": [],
              "scope": "explicit_internal_model_tensor_point", "closed_kernel_internals": False}
    debugger = PrecisionDebugger(config={"output_dir": str(root), "run_id": "capture", "mode": "tensor"})
    try:
        debugger.start()
        if args.mode == "compile":
            if args.device == "musa":
                import torchada
            from torch._dynamo.backends.registry import lookup_backend
            actual_backend = lookup_backend("inductor" if args.device == "musa" else "aot_eager")
            compiled = []
            def backend(gm, inputs):
                compiled.append(True)
                return actual_backend(gm, inputs)
            execute = torch.compile(model, backend=backend, fullgraph=True)
        else:
            if args.device != "musa":
                raise ValueError("This example requires MUSA Graph")
            api = torch.musa
            stream = api.Stream()
            stream.wait_stream(api.current_stream())
            with api.stream(stream), torch.no_grad():
                for _ in range(3):
                    model(initial)
            api.current_stream().wait_stream(stream)
            graph = api.MUSAGraph()
            with api.graph(graph), torch.no_grad():
                output = model(initial)
        for step in range(3):
            value = torch.arange(8, dtype=torch.float32, device=args.device).reshape(2, 4) + step * 0.5
            with torch.no_grad():
                expected_point = reference(value)
                expected_output = torch.nn.functional.gelu(expected_point)
                if args.mode == "compile":
                    output = execute(value)
                    assert compiled, "No actual compiler backend call"
                else:
                    initial.copy_(value)
                    graph.replay()
                assert torch.allclose(expected_output, output, atol=1e-5, rtol=1e-4)
                assert torch.allclose(expected_point, point.snapshot_buffer, atol=1e-5, rtol=1e-4)
            debugger.set_context(step=step)
            generation = point.save(debugger, "linear.internal", execution_mode="compile" if args.mode == "compile" else "musa_graph")
            result["generations"].append(generation)
            try:
                point.snapshot()
                raise AssertionError("Stale graph snapshots must be rejected")
            except RuntimeError:
                pass
        assert len(set(result["generations"])) == 3
        result["status"] = "pass"
    finally:
        debugger.close()
        (root / "graph_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
