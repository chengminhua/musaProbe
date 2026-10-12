"""Explicit observation of synchronous distributed collectives.

Every participant must use the same stable group_id and sequence. This wrapper
does not guess identities from tensor values or wait on asynchronous work.
"""
from __future__ import annotations


def observed_all_reduce(debugger, tensor, *, group_id, sequence, group=None, op=None):
    import torch.distributed as dist
    if not isinstance(group_id, str) or not group_id or type(sequence) is not int or sequence < 0:
        raise ValueError("Provide a nonempty group_id and nonnegative collective sequence")
    if not dist.is_initialized():
        raise RuntimeError("A distributed process group must already be initialized")
    participants = (list(range(dist.get_world_size())) if group is None
                    else dist.get_process_group_ranks(group))
    reduction = dist.ReduceOp.SUM if op is None else op
    metadata = {"operation": "all_reduce", "group_id": group_id, "sequence": sequence,
                "participants": participants, "reduction": str(reduction), "synchronous": True}
    # phase() restores all context fields even on collective failure.
    with debugger.phase(debugger._context["phase"]):
        debugger.set_context(communication={**metadata, "stage": "input"})
        debugger.save(tensor, f"communication.{group_id}.{sequence}", kind="communication_input")
        try:
            dist.all_reduce(tensor, op=reduction, group=group, async_op=False)
        except BaseException as error:
            reason = f"observed all_reduce failed: {type(error).__name__}: {error}"
            debugger.errors.append(reason)
            debugger._notice(dict(debugger._context, module=f"communication.{group_id}.{sequence}",
                                  communication={**metadata, "stage": "output"}, call_index=0, op=None),
                             "communication_output", "value", "error", reason)
            raise
        debugger.set_context(communication={**metadata, "stage": "output"})
        debugger.save(tensor, f"communication.{group_id}.{sequence}", kind="communication_output")
    return tensor
