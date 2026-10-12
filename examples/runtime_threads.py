"""Bounded per-rank runtime/thread diagnostic; launch explicitly with torchrun.

Writes only beneath --output. No model or training is loaded. By default runs
one scalar collective, with optional earlier stop points for controlled tests.
It does not change container limits, environment variables or system settings.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import traceback


def thread_snapshot(proc_root=Path("/proc/self")):
    """Read this process only; no ptrace, sampling threads, or privilege changes."""
    groups={}; errors=[]; entries=sorted((proc_root/"task").iterdir(),key=lambda p:int(p.name))
    for task in entries[:4096]:
        try:
            name=(task/"comm").read_text().strip()
            try: wait_channel=(task/"wchan").read_text().strip()
            except OSError as exc: wait_channel=f"unavailable:{type(exc).__name__}"
            key=(name,wait_channel)
            group=groups.setdefault(key,{"comm":name,"wchan":wait_channel,"count":0,"sample_tids":[]})
            group["count"]+=1
            if len(group["sample_tids"])<3: group["sample_tids"].append(int(task.name))
        except FileNotFoundError:
            # Threads may exit while /proc is read; this is not a consistent kernel snapshot.
            continue
        except OSError as exc:
            errors.append({"tid":int(task.name),"error":f"{type(exc).__name__}: {exc}"})
    libraries=set()
    for line in (proc_root/"maps").read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[5].startswith("/") and ".so" in Path(fields[5]).name:
            libraries.add(fields[5])
    return {"thread_groups":sorted(groups.values(),key=lambda g:(-g["count"],g["comm"],g["wchan"])),
            "thread_snapshot_truncated":len(entries)>4096,"thread_read_errors":errors}, libraries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("musa","cuda"),default="musa")
    parser.add_argument("--intra-threads",type=int,default=2)
    parser.add_argument("--interop-threads",type=int,default=1)
    parser.add_argument("--stop-after",choices=("import","set_device","tensor","init_dist","all_reduce"),default="all_reduce")
    parser.add_argument("--timeout",type=int,default=45)
    args=parser.parse_args()
    if not 1<=args.intra_threads<=32 or not 1<=args.interop_threads<=32 or not 1<=args.timeout<=120:
        parser.error("Thread counts must be 1..32 and timeout 1..120 seconds")
    if not all(name in os.environ for name in ("RANK","LOCAL_RANK","WORLD_SIZE")):
        parser.error("Launch with torchrun so rank/device/collective scope is explicit")
    rank,local_rank,world=(int(os.environ[k]) for k in ("RANK","LOCAL_RANK","WORLD_SIZE"))
    root=Path(args.output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    stream=(root/f"rank_{rank:05d}.jsonl").open("x",buffering=1)
    started=time.monotonic();initialized=False;api=None;success=False;prior_libraries=set()

    def record(stage,**values):
        nonlocal prior_libraries
        fields={}
        for line in Path("/proc/self/status").read_text().splitlines():
            key,_,value=line.partition(":")
            if key in ("Threads","VmRSS","VmSize"):fields[key]=value.strip()
        cgroup={}
        for name in ("pids.current","pids.max","pids.events"):
            path=Path("/sys/fs/cgroup")/name
            if path.is_file():cgroup[name]=path.read_text().strip()
        detail,libraries=thread_snapshot()
        newly_loaded=sorted(libraries-prior_libraries)
        unloaded=sorted(prior_libraries-libraries)
        prior_libraries=libraries
        event={"rank":rank,"local_rank":local_rank,"world_size":world,"pid":os.getpid(),
               "stage":stage,"elapsed_seconds":time.monotonic()-started,
               "affinity_cpu_count":len(os.sched_getaffinity(0)),"reported_cpu_count":os.cpu_count(),"process":fields,
               "container_cgroup":cgroup,"newly_loaded_libraries":newly_loaded,
               "unloaded_libraries":unloaded,"loaded_library_count":len(libraries),**detail,**values}
        stream.write(json.dumps(event)+"\n");stream.flush()
        print(json.dumps(event),flush=True)

    try:
        record("before_torch_import")
        import torch
        record("after_torch_import",intra_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        torch.set_num_interop_threads(args.interop_threads);torch.set_num_threads(args.intra_threads)
        record("after_thread_configuration",intra_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        if args.device=="musa":import torch_musa  # noqa: F401
        api=getattr(torch,args.device)
        record("after_backend_import")
        if args.stop_after=="import":success=True;return 0
        record("before_set_device");api.set_device(local_rank);record("after_set_device")
        if args.stop_after=="set_device":success=True;return 0
        record("before_tensor")
        tensor=torch.tensor([float(rank+1)],device=f"{args.device}:{local_rank}")
        api.synchronize();record("after_tensor")
        if args.stop_after=="tensor":success=True;return 0
        if torch.distributed.is_initialized():raise RuntimeError("Refusing an existing process group")
        record("before_init_dist")
        torch.distributed.init_process_group("mccl" if args.device=="musa" else "nccl",timeout=timedelta(seconds=args.timeout))
        initialized=True;record("after_init_dist")
        if args.stop_after=="init_dist":success=True;return 0
        record("before_all_reduce")
        torch.distributed.all_reduce(tensor)
        actual=float(tensor.item());expected=float(world*(world+1)//2)
        record("after_all_reduce",actual=actual,expected=expected)
        if actual!=expected:raise RuntimeError("Scalar all_reduce returned an unexpected value")
        success=True;return 0
    except BaseException as exc:
        record("error",exception=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        return 2
    finally:
        if initialized:
            try:
                # No barrier in the failure path. Release only this diagnostic's group.
                if success:api.synchronize()
                torch.distributed.destroy_process_group();record("after_destroy_process_group")
            except BaseException as exc:
                success=False;record("cleanup_error",exception=f"{type(exc).__name__}: {exc}")
        record("finished",status="pass" if success else "error");stream.close()
        if not success:raise SystemExit(2)


if __name__=="__main__":raise SystemExit(main())
