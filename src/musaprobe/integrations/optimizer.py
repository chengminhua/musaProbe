"""Explicit state observation for ordinary torch optimizers at caller boundaries."""
from __future__ import annotations


def capture_optimizer_state(debugger, model, optimizer, *, stage="post_step"):
    """Capture tensor states whose optimizer parameter is the actual model object.

    No ordering-based parameter guessing is used for master weights or sharded
    optimizers. Such optimizers need an adapter with a proven ownership mapping.
    States are observed after the caller's completed step, never materialized by
    performing an optimizer update.
    """
    import torch
    if stage not in ("pre_step", "post_step"):
        raise ValueError("stage must be pre_step or post_step")
    if not isinstance(optimizer, torch.optim.Optimizer):
        return {"status": "unsupported", "reason": "Optimizer requires an explicit parameter/state ownership adapter"}
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    unknown = [parameter for parameter in optimizer.state if id(parameter) not in names]
    if unknown:
        return {"status": "unsupported", "reason": "Optimizer state contains master/sharded parameters without a verified model mapping"}
    captured, unsupported, skipped = [], [], []
    pending = False
    with debugger.phase("optimizer"):
        debugger.set_context(optimizer_type=type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                             state_stage=stage)
        for parameter, states in optimizer.state.items():
            name = names[id(parameter)]
            for key, value in states.items():
                if not isinstance(key, str):
                    unsupported.append({"parameter": name, "reason": "Non-string state key"})
                    continue
                if not isinstance(value, torch.Tensor):
                    unsupported.append({"parameter": name, "state": key, "reason": "Non-tensor optimizer metadata"})
                    continue
                module = name + "." + key
                identity = ("optimizer_adapter", debugger._context["step"], debugger._context["microbatch"], stage, module)
                metadata = dict(debugger._context, module=module, op=None,
                                call_index=debugger._counts.get(identity, 0), stack=debugger._stack())
                if not debugger._selected(metadata):
                    skipped.append({"parameter": name, "state": key, "reason": "capture policy excluded this state"})
                    continue
                debugger._counts[identity] += 1
                event = debugger._write(value, metadata, "optimizer_state", "value")
                if event is None or event["state"] not in ("tensor", "statistics", "pending"):
                    unsupported.append({"parameter": name, "state": key,
                                        "reason": event.get("reason") if event else "collector did not produce an event"})
                    continue
                pending = pending or event["state"] == "pending" or debugger.config.async_write
                captured.append({"parameter": name, "state": key, "shape": list(value.shape)})
    status = ("incomplete" if unsupported else "skipped" if skipped and not captured else
              "incomplete" if not captured else "submitted" if pending else "pass")
    return {"status": status, "captured": captured, "unsupported": unsupported,
            "skipped": skipped, "stage": stage}
