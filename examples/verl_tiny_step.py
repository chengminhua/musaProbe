"""One real local VeRL HF rollout and PPO actor update on a tiny model.

Run in a dedicated process/container. This does not start Ray, an HTTP server,
or distributed RL workers. The reward is synthetic, not a quality benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import traceback


def assert_unhooked_model(model):
    """The control model must never inherit probe hooks through deepcopy."""
    for name,module in model.named_modules():
        for attribute in ("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks"):
            if getattr(module,attribute,None):
                raise RuntimeError(f"Baseline module has hooks: {name}.{attribute}")
    for name,parameter in model.named_parameters():
        if getattr(parameter,"_backward_hooks",None) or getattr(parameter,"_post_accumulate_grad_hooks",None):
            raise RuntimeError(f"Baseline parameter has hooks: {name}")


def _equal_tree(left,right):
    import numpy as np
    import torch
    if type(left) is not type(right): return False
    if isinstance(left,torch.Tensor):
        return left.device==right.device and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return left.dtype==right.dtype and left.shape==right.shape and np.array_equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_equal_tree(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(_equal_tree(a,b) for a,b in zip(left,right))
    return left==right


def clone_ppo_batch(batch):
    """Clone numeric inputs without DataProto's device-changing pickle protocol."""
    import torch
    from tensordict import TensorDict
    if type(batch.batch) is not TensorDict or len(batch.batch.batch_size)!=1:
        raise TypeError("Paired PPO supports an ordinary TensorDict with exactly one batch dimension")
    if not batch.batch.keys(): raise ValueError("Paired PPO input must contain tensors")
    tensors={}
    for name,value in batch.batch.items():
        if (not isinstance(name,str) or type(value) is not torch.Tensor or value.is_nested or
                value.layout!=torch.strided or value.requires_grad):
            raise TypeError("Paired PPO input must contain flat, dense, detached ordinary tensors")
        # TensorDict.device may be None although its values are MUSA tensors.
        # Cloning the values preserves their devices without serializing them.
        tensors[name]=value.detach().clone()
    tensor_dict=TensorDict(source=tensors,batch_size=batch.batch.batch_size,device=batch.batch.device,
                           names=batch.batch.names)
    cloned=type(batch)(batch=tensor_dict,meta_info=copy.deepcopy(batch.meta_info),
                       non_tensor_batch=copy.deepcopy(batch.non_tensor_batch))
    if (type(cloned) is not type(batch) or type(cloned.batch) is not type(batch.batch) or
            cloned.batch.batch_size!=batch.batch.batch_size or cloned.batch.device!=batch.batch.device or
            cloned.batch.names!=batch.batch.names or set(batch.batch.keys())!=set(cloned.batch.keys()) or
            not _equal_tree(batch.meta_info,cloned.meta_info) or
            not _equal_tree(batch.non_tensor_batch,cloned.non_tensor_batch)):
        raise RuntimeError("PPO input clone changed DataProto/TensorDict fields")
    for name,value in batch.batch.items():
        other=cloned.batch[name]
        if (type(other) is not type(value) or value.device!=other.device or not _equal_tree(value,other) or
                (value.numel() and value.untyped_storage().data_ptr()==other.untyped_storage().data_ptr())):
            raise RuntimeError(f"PPO input is changed or shares storage: {name}")
    return cloned


def snapshot_rng(device):
    """Default generators only; no claim about private framework generators."""
    import numpy as np
    import torch
    accelerator=[]
    if device!="cpu":
        api=getattr(torch,device)
        api.synchronize()
        accelerator=[value.cpu().clone() for value in api.get_rng_state_all()]
        if len(accelerator)!=api.device_count():
            raise RuntimeError("RNG snapshot did not include every visible accelerator")
    return {"python":copy.deepcopy(random.getstate()),"numpy":copy.deepcopy(np.random.get_state()),
            "torch_cpu":torch.get_rng_state().clone(),"device":device,"accelerator":accelerator}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["device"]!="cpu":
        api=getattr(torch,state["device"])
        if len(state["accelerator"])!=api.device_count():
            raise RuntimeError("Visible accelerator count changed during paired PPO")
        api.set_rng_state_all(state["accelerator"])


def _save_tensor(path,value):
    import numpy as np
    import torch
    value=value.detach().cpu().contiguous()
    try: array=value.numpy();encoding="numpy"
    except TypeError: array=value.reshape(-1).view(torch.uint8).numpy();encoding="raw_bytes"
    np.save(path,array,allow_pickle=False)
    return {"file":path.name,"shape":list(value.shape),"dtype":str(value.dtype),
            "stored_dtype":str(array.dtype),"encoding":encoding}


def save_rng(path,state):
    """Persist replayable numeric arrays and JSON, without pickle."""
    import numpy as np
    path.mkdir()
    np.save(path/"numpy_keys.npy",state["numpy"][1],allow_pickle=False)
    result={"python":state["python"],"numpy":{"algorithm":state["numpy"][0],"keys":"numpy_keys.npy",
            "position":state["numpy"][2],"has_gauss":state["numpy"][3],"cached_gaussian":state["numpy"][4]},
            "device":state["device"],"torch_cpu":_save_tensor(path/"torch_cpu.npy",state["torch_cpu"]),
            "accelerator":[_save_tensor(path/f"device_{i}.npy",value) for i,value in enumerate(state["accelerator"])]}
    (path/"state.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


def compare_ppo_parameters(model,baseline_model,output):
    import torch
    output.mkdir()
    observed=dict(model.named_parameters());baseline=dict(baseline_model.named_parameters())
    if observed.keys()!=baseline.keys(): raise RuntimeError("Paired PPO parameter names differ")
    records=[]
    for index,(name,parameter) in enumerate(observed.items()):
        left=parameter.detach().cpu();right=baseline[name].detach().cpu()
        if left.shape!=right.shape or left.dtype!=right.dtype:
            raise RuntimeError(f"Paired PPO parameter shape/dtype differ: {name}")
        finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        records.append({"name":name,"exact_equal":bool(torch.equal(left,right)),"finite":finite,
                        "different_elements":int(torch.count_nonzero(left!=right)),
                        "max_abs":float((left.double()-right.double()).abs().max()) if finite and left.numel() else None,
                        "captured":_save_tensor(output/f"captured_{index:04d}.npy",left),
                        "baseline":_save_tensor(output/f"baseline_{index:04d}.npy",right)})
    report={"all_exact":bool(records) and all(item["exact_equal"] for item in records),
            "all_finite":bool(records) and all(item["finite"] for item in records),
            "parameter_tensors":len(records),"parameters":records}
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def validate_actor_capture(capture_dir, device):
    """Require real, complete actor forward/backward files before declaring pass."""
    import numpy as np
    root=Path(capture_dir).resolve()
    manifests=list(root.glob("rank_*/manifest.json"))
    if len(manifests)!=1: raise RuntimeError("Expected one complete local actor capture rank")
    manifest_path=manifests[0];rank=manifest_path.parent
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get("status")!="complete" or manifest.get("rank")!=0 or
            manifest.get("world_size")!=1 or manifest.get("event_budget_exhausted")):
        raise RuntimeError("Actor capture manifest is incomplete or exhausted")
    events_path=rank/"events.jsonl"
    if events_path.stat().st_size>8*1024**2: raise RuntimeError("Actor event read budget exceeded")
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    if not 4<=len(events)<=200 or manifest.get("events_count")!=len(events):
        raise RuntimeError("Actor capture event count is missing or inconsistent")
    paths=set();total=0;counts={};forward={};gradients=[];nonzero_gradients=0
    for index,event in enumerate(events):
        if (event.get("event_id")!=index or event.get("state")!="tensor" or
                event.get("module")!="model.layers.0.mlp.down_proj" or
                not str(event.get("device","")).startswith(device+":")):
            raise RuntimeError("Unexpected actor event identity, state or device")
        relative=event.get("file")
        if not isinstance(relative,str) or Path(relative).is_absolute():
            raise RuntimeError("Actor tensor file must be relative")
        path=(rank/relative).resolve()
        if not path.is_relative_to(rank) or not path.is_file() or path in paths:
            raise RuntimeError("Actor tensor file is missing, duplicated or outside its rank")
        total+=path.stat().st_size
        if total>64*1024**2 or path.stat().st_size!=event.get("bytes"):
            raise RuntimeError("Actor tensor byte count is inconsistent or exceeds budget")
        array=np.load(path,allow_pickle=False,mmap_mode="r")
        if (list(array.shape)!=event.get("shape") or str(array.dtype)!=event.get("stored_dtype") or
                array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
            raise RuntimeError("Actor tensor dtype, shape or finite-value verification failed")
        paths.add(path);kind=event.get("kind");counts[kind]=counts.get(kind,0)+1
        identity=(event.get("step"),event.get("microbatch"),event.get("module"),event.get("call_index"),event.get("tensor_path"))
        if kind in ("input","output") and event.get("phase")=="forward":
            forward[(identity,kind)]=event["shape"]
        elif kind in ("grad_input","grad_output") and event.get("phase")=="backward":
            gradients.append((identity,kind.removeprefix("grad_"),event["shape"]))
            nonzero_gradients+=int(np.count_nonzero(array))
        else:raise RuntimeError("Actor capture has an unexpected kind/phase")
    if not all(counts.get(kind,0)>0 for kind in ("input","output","grad_input","grad_output")):
        raise RuntimeError("Actor capture lacks forward or backward boundary tensors")
    if any(forward.get((identity,kind))!=shape for identity,kind,shape in gradients):
        raise RuntimeError("Actor gradient is not linked to its captured forward boundary")
    if not nonzero_gradients: raise RuntimeError("Actor capture has no nonzero gradient observation")
    if total!=manifest.get("tensor_bytes") or paths!={p.resolve() for p in rank.rglob("*.npy")}:
        raise RuntimeError("Actor capture byte total or referenced file inventory is inconsistent")
    return {"manifest_status":"complete","event_count":len(events),"kinds":counts,
            "tensor_files_verified":len(paths),"tensor_bytes":total,"nonzero_gradient_elements":nonzero_gradients,
            "gradient_forward_links_verified":True}


def run(model_dir, output_dir, device="musa"):
    root=Path(output_dir).expanduser().resolve(); root.mkdir(parents=True,exist_ok=False)
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"): os.environ[key]="1"
    for key,leaf in (("XDG_CACHE_HOME","cache"),("TORCHINDUCTOR_CACHE_DIR","inductor"),
                     ("TRITON_CACHE_DIR","triton"),("HF_HOME","huggingface")):
        os.environ.setdefault(key,str(root/leaf))
    result={"status":"unsupported","scope":"single_process_hf_rollout_and_ppo_actor_update",
            "device":device,"distributed_rl_verified":False,"backend_execution_verified":False,
            "rng_transparency_verified":False,"uncaptured_PPO_baseline_compared":False,
            "reward":"synthetic_positive_mean_generated_token_id","phase":"imports"}
    initialized=False;rollout=None
    try:
        import numpy as np
        import torch
        if device=="musa": import torch_musa  # noqa: F401
        if device not in ("musa","cuda") or not getattr(torch,device).is_available():
            raise RuntimeError("This example requires one explicitly selected available GPU")
        api=getattr(torch,device); api.set_device(0)
        from transformers import AutoConfig,AutoModelForCausalLM
        from verl import DataProto,__version__ as verl_version
        from verl.utils.device import get_device_name
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from verl.workers.config.actor import FSDPActorConfig
        from verl.workers.config.engine import FSDPEngineConfig
        from verl.workers.config import HFModelConfig,RolloutConfig
        from musaprobe import PrecisionDebugger
        from musaprobe.integrations.verl import create_local_hf_rollout
        result.update(verl_version=verl_version,observed_backend=get_device_name())
        if get_device_name()!=device: raise RuntimeError("VeRL did not select the requested backend")
        if torch.distributed.is_initialized():
            raise RuntimeError("Refusing to reuse an existing distributed process group")
        result["phase"]="model_load"
        source=Path(model_dir).expanduser().resolve()
        if not source.is_dir(): raise ValueError("A local tiny model directory is required")
        config=AutoConfig.from_pretrained(source,local_files_only=True,trust_remote_code=False)
        if (config.model_type!="llama" or not 0<config.vocab_size<=1024 or
                not 0<config.hidden_size<=256 or not 0<config.num_hidden_layers<=4 or
                not 0<config.intermediate_size<=1024):
            raise ValueError("Only a bounded tiny Llama configuration is accepted")
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                   torch_dtype=torch.float32,attn_implementation="eager").to(device)
        if sum(p.numel() for p in model.parameters())>2_000_000:
            raise ValueError("The tiny model budget is two million parameters")
        # VeRL actor calls get_rank even for an ordinary non-FSDP module. This
        # private, one-rank Gloo group carries no GPU gradient communication.
        torch.distributed.init_process_group("gloo",init_method=(root/"process_group").as_uri(),
                                             rank=0,world_size=1,timeout=timedelta(seconds=60))
        initialized=True
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
        actor_config=FSDPActorConfig(ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=2,
                    rollout_n=1,ppo_epochs=1,use_torch_compile=False,use_remove_padding=False,
                    use_fused_kernels=False,use_dynamic_bsz=False,calculate_entropy=False,
                    fsdp_config=FSDPEngineConfig(dtype="bfloat16"))
        actor=DataParallelPPOActor(actor_config,model,optimizer)
        assert_unhooked_model(model)
        # One deepcopy transaction preserves the copied optimizer-to-Parameter
        # links. This occurs before installing any capture hook.
        baseline_model,baseline_optimizer=copy.deepcopy((model,optimizer))
        assert_unhooked_model(baseline_model)
        if ({id(p) for group in baseline_optimizer.param_groups for p in group["params"]}!=
                {id(p) for p in baseline_model.parameters()} or
                any(a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
                    for a,b in zip(model.parameters(),baseline_model.parameters())) or
                not _equal_tree(model.state_dict(),baseline_model.state_dict()) or
                not _equal_tree(optimizer.state_dict(),baseline_optimizer.state_dict())):
            raise RuntimeError("Baseline model/optimizer are not independent identical copies")
        baseline_actor=DataParallelPPOActor(copy.deepcopy(actor_config),baseline_model,baseline_optimizer)
        # Rollout owns a separate tiny module. Its real offload/resume must not
        # move the actor or invalidate the optimizer's Parameter references.
        rollout_module=copy.deepcopy(model)
        if any(a.data_ptr()==b.data_ptr() for a,b in zip(model.parameters(),rollout_module.parameters())):
            raise RuntimeError("Actor and rollout unexpectedly share parameter storage")
        # Current VeRL dataclasses retain only mode="async". The local HF
        # adapter still calls generate_sequences directly; no server is started.
        rollout_config=RolloutConfig(name="hf",do_sample=False,temperature=1.0,
                    response_length=4,top_p=1.0,top_k=0,tensor_model_parallel_size=1,
                    prompt_length=3,max_model_len=64,max_num_seqs=2,
                    skip_dump_dir=str(root/"rollout_dump"))
        hf_config=HFModelConfig(path=str(source),load_tokenizer=False,trust_remote_code=False,
                    enable_gradient_checkpointing=False,use_remove_padding=False,
                    override_config={"attn_implementation":"eager"})
        rollout=create_local_hf_rollout(rollout_module,rollout_config,hf_config,device=device)
        result["rollout_configuration"]=rollout.hf_config_evidence
        initial_actor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        asyncio.run(rollout.release())
        if any(p.device.type!=device for p in model.parameters()):
            raise RuntimeError("Releasing rollout moved actor parameters")
        asyncio.run(rollout.resume(["weights","kv_cache"]))
        asyncio.run(rollout.update_weights(iter(model.state_dict().items())))
        if any(not torch.equal(initial_actor[n],p.detach().cpu()) for n,p in model.named_parameters()):
            raise RuntimeError("Rollout lifecycle changed actor parameters")
        prompt_ids=torch.tensor([[1,4,5],[1,6,7]],device=device)
        prompts=DataProto.from_dict(tensors={"input_ids":prompt_ids,
                    "attention_mask":torch.ones_like(prompt_ids),
                    "position_ids":torch.arange(3,device=device).expand(2,-1)},
                    meta_info={"eos_token_id":2,"pad_token_id":3})
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        probe=PrecisionDebugger(config={"output_dir":str(root/"captures"),"run_id":"verl_step",
                     "mode":"tensor","modules":["model.layers.0.mlp.down_proj"],
                     "max_events":200,"max_rank_bytes":64*1024**2},
                     metadata={"framework":"verl","scope":result["scope"]})
        with probe:
            probe.set_context(step=0,microbatch=0,phase="forward"); probe.start(model)
            result["phase"]="rollout"
            batch=rollout.generate_sequences(prompts)
            batch.meta_info.update(micro_batch_size=2,temperature=1.0,use_dynamic_bsz=False,pad_token_id=3)
            response_length=batch.batch["responses"].shape[-1]
            batch.batch["response_mask"]=batch.batch["attention_mask"][:,-response_length:]
            result["phase"]="old_log_probs"
            batch.batch["old_log_probs"]=actor.compute_log_prob(batch)["log_probs"].detach()
            rewards=1+batch.batch["responses"].float().mean(dim=-1)/config.vocab_size
            batch.batch["advantages"]=rewards[:,None].expand(-1,response_length)*batch.batch["response_mask"]
            for name in ("responses","old_log_probs","advantages","response_mask"):
                value=batch.batch[name].detach().cpu()
                if value.is_floating_point(): value=value.float()
                np.save(root/(name+".npy"),value.numpy(),allow_pickle=False)
            baseline_batch=clone_ppo_batch(batch)
            paired=root/"ppo_transparency";paired.mkdir()
            inputs=paired/"inputs";inputs.mkdir()
            input_files={name:_save_tensor(inputs/f"tensor_{index:03d}.npy",value)
                         for index,(name,value) in enumerate(batch.batch.items())}
            (inputs/"fields.json").write_text(json.dumps({"tensors":input_files,"meta_info":batch.meta_info,
                          "non_tensor_keys":list(batch.non_tensor_batch)},indent=2,allow_nan=False)+"\n")
            # compute_log_prob switches the actor to eval. Mirror every module
            # flag even though current update_policy itself calls train().
            for name,module in model.named_modules():
                baseline_model.get_submodule(name).training=module.training
            assert_unhooked_model(baseline_model)
            if (not _equal_tree(model.state_dict(),baseline_model.state_dict()) or
                    not _equal_tree(optimizer.state_dict(),baseline_optimizer.state_dict())):
                raise RuntimeError("Actor state changed before paired PPO initial-state alignment")
            rng_before=snapshot_rng(device)
            result["phase"]="ppo_update"
            # update_policy owns both its forward and backward passes. Do not
            # falsely label the entire method as a backward/recompute context.
            metrics=actor.update_policy(batch)
            rng_captured_after=snapshot_rng(device)
            probe.stop()
        if probe.errors: raise RuntimeError("musaProbe capture recorded errors: "+str(probe.errors))
        result["actor_capture_validation"]=validate_actor_capture(root/"captures"/"verl_step",device)
        result["phase"]="uncaptured_ppo_baseline"
        assert_unhooked_model(model);assert_unhooked_model(baseline_model)
        try:
            restore_rng(rng_before)
            baseline_metrics=baseline_actor.update_policy(baseline_batch)
            rng_baseline_after=snapshot_rng(device)
        finally:
            # The owned actor and its subsequent rollout keep the captured
            # trajectory; only the independent control used the restored RNG.
            restore_rng(rng_captured_after)
        assert_unhooked_model(baseline_model)
        for label,state in (("before",rng_before),("captured_after",rng_captured_after),
                            ("baseline_after",rng_baseline_after)):
            save_rng(paired/label,state)
        parameter_comparison=compare_ppo_parameters(model,baseline_model,paired/"parameters")
        rng_comparison={name:_equal_tree(rng_captured_after[name],rng_baseline_after[name])
                        for name in ("python","numpy","torch_cpu","device","accelerator")}
        result.update(uncaptured_PPO_baseline_compared=True,
                      rng_transparency_verified=all(rng_comparison.values()),
                      parameter_transparency_verified=parameter_comparison["all_exact"] and parameter_comparison["all_finite"],
                      ppo_transparency={"baseline_has_no_hooks":True,"inputs_equal_and_storage_independent":True,
                          "parameter_tensors":parameter_comparison["parameter_tensors"],
                          "parameters_all_finite":parameter_comparison["all_finite"],
                          "rng_exact":rng_comparison,"baseline_metrics":baseline_metrics,
                          "evidence":"ppo_transparency",
                          "scope":"one PPO update; Python/NumPy default RNG, torch CPU and all visible accelerator default generators; no private framework generators"})
        if not result["rng_transparency_verified"] or not result["parameter_transparency_verified"]:
            result["status"]="mismatch"
            raise RuntimeError("Captured versus unhooked PPO differs exactly; see ppo_transparency evidence")
        api.synchronize()
        deltas={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in model.named_parameters()}
        changed=sum(value>0 for value in deltas.values())
        if not changed or not all(np.isfinite(value) for value in deltas.values()):
            raise RuntimeError("PPO update did not produce a finite parameter change")
        result["phase"]="rollout_weight_refresh"
        asyncio.run(rollout.update_weights(iter(model.state_dict().items())))
        asyncio.run(rollout.release())
        asyncio.run(rollout.resume(["weights","kv_cache"]))
        refreshed=rollout.generate_sequences(prompts)
        np.save(root/"refreshed_responses.npy",refreshed.batch["responses"].detach().cpu().numpy(),allow_pickle=False)
        result.update(status="pass",phase="complete",backend_execution_verified=True,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      changed_parameters=changed,max_parameter_change=max(deltas.values()),
                      metrics=metrics,capture=str(root/"captures"/"verl_step"),
                      rollout_scope="owned_local_HF_model_with_real_release_resume_full_weight_refresh",
                      actor_rollout_storage_shared=False,
                      gradient_capture_scope="module_boundary_backward_only; actor clears parameter gradients")
    except Exception as exc:
        result.update(status="mismatch" if result["status"]=="mismatch" else "unsupported",
                      reason=f"{type(exc).__name__}: {exc}")
        (root/"traceback.txt").write_text(traceback.format_exc())
    finally:
        if rollout is not None:
            try:
                asyncio.run(rollout.release())
                result["rollout_lifecycle"]=rollout.lifecycle_events
            except Exception as exc:
                result.update(status="error",rollout_cleanup_error=f"{type(exc).__name__}: {exc}")
        if initialized:
            try:
                try: getattr(torch,device).synchronize()
                finally: torch.distributed.destroy_process_group()
            except Exception as exc: result.update(status="error",cleanup_error=f"{type(exc).__name__}: {exc}")
        (root/"verl_step.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("musa","cuda"),default="musa")
    args=parser.parse_args();result=run(args.model,args.output,args.device)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="pass" else 2)
