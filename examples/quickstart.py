#!/usr/bin/env python3
"""Generate real captures, compare CPU/MUSA and locate a controlled defect."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from musaprobe import PrecisionDebugger
from musaprobe.compare import compare_runs
from musaprobe.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "musa"), default="musa")
    args = parser.parse_args()
    if args.device == "musa":
        import torch_musa  # noqa: F401
        assert torch.musa.is_available(), "A MUSA device is required"
    torch.manual_seed(42)
    base = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    data = torch.randn(3, 8)
    target = torch.randn(3, 4)
    run_group = args.output.resolve() / datetime.now(timezone.utc).strftime("demo_%Y%m%dT%H%M%S_%fZ")
    run_group.mkdir(parents=True)

    def run(device, name, inject=False):
        model = copy.deepcopy(base).to(device)
        x = data.to(device).detach().requires_grad_()
        y = target.to(device)
        hook = None
        if inject:
            hook = model[0].register_forward_hook(lambda m, inputs, output: output + 0.1)
        def execute():
            model.zero_grad(set_to_none=True)
            x.grad = None
            prediction = model(x)
            loss = (prediction-y).square().mean()
            loss.backward()
            return {"output": prediction.detach().cpu().clone(), "loss":loss.detach().cpu().clone(),
                    "input_grad":x.grad.detach().cpu().clone(),
                    **{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}}
        original = execute()
        model.zero_grad(set_to_none=True)
        x.grad = None
        probe = PrecisionDebugger(config={"output_dir":str(run_group), "run_id":name,
                                           "mode":"tensor", "microbatches":[0],
                                           "apis":["aten.add.Tensor","aten.mm.default","aten.relu.default"]})
        try:
            with probe:
                probe.set_context(step=0,microbatch=0,phase="forward")
                probe.start(model)
                prediction=model(x)
                loss=(prediction-y).square().mean()
                with probe.phase("backward"):
                    loss.backward()
                probe.capture_parameter_grads(model)
                probe.stop()
                captured={"output":prediction.detach().cpu().clone(),"loss":loss.detach().cpu().clone(),
                          "input_grad":x.grad.detach().cpu().clone(),
                          **{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}}
                for key in original:
                    torch.testing.assert_close(original[key],captured[key],atol=0,rtol=0)
                cleaned=execute()
                for key in original:
                    torch.testing.assert_close(original[key],cleaned[key],atol=0,rtol=0)
        finally:
            if hook is not None:
                hook.remove()
        return run_group/name

    cpu=run("cpu","reference_cpu")
    candidate=run(args.device,"candidate")
    injected=run(args.device,"injected",True)
    normal=compare_runs(cpu,candidate,output=run_group/"comparison_normal")
    defect=compare_runs(cpu,injected,output=run_group/"comparison_injected")
    assert normal["overall"]=="pass", normal["summary"]
    assert defect["overall"]=="fail", defect["summary"]
    first=defect.get("first_difference") or {}
    assert first.get("identity",{}).get("module")=="0", first
    summary={"normal":normal["overall"],"injected":defect["overall"],
             "first_difference":first,"capture_transparency":"exact output/loss/input+parameter gradients",
             "reports":[str(run_group/"comparison_normal/report.html"),str(run_group/"comparison_injected/report.html")],
             "scope":"single-device eager float32 small network; controlled injection, not an observed hardware bug"}
    atomic_json(run_group/"demo_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
