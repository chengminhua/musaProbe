"""Explicit, single-process lifecycle for an owned VeRL HF rollout model.

VeRL is imported only by ``create_local_hf_rollout``. This adapter does not
register a Ray/server rollout, implement FSDP, or move an actor's model. Pass a
separate model whose parameter storage is not shared with an actor/optimizer.
"""
from __future__ import annotations


def _validate_hf_config(config):
    """Validate every configuration field read by the target legacy HF path."""
    if not callable(getattr(config,"get",None)):
        raise TypeError("Use the installed VeRL RolloutConfig with its Mapping.get interface")
    if getattr(config,"mode",None)!="async":
        raise ValueError("The supported VeRL dataclass contract uses mode='async'; local HF execution remains a direct call")
    if type(getattr(config,"do_sample",None)) is not bool:
        raise ValueError("RolloutConfig.do_sample must be a bool")
    if type(getattr(config,"response_length",None)) is not int or config.response_length<=0:
        raise ValueError("RolloutConfig.response_length must be a positive integer")
    temperature=getattr(config,"temperature",None)
    if not isinstance(temperature,(float,int)) or isinstance(temperature,bool) or temperature<=0:
        raise ValueError("RolloutConfig.temperature must be positive")
    top_p=config.get("top_p",1.0);top_k=config.get("top_k",0)
    if not isinstance(top_p,(int,float)) or not 0<top_p<=1 or type(top_k) is not int or top_k < -1:
        raise ValueError("Invalid HF top_p/top_k configuration")
    missing=object();micro_batch=config.get("micro_batch_size",missing)
    if micro_batch is not missing and (type(micro_batch) is not int or micro_batch<=0):
        raise ValueError("Legacy HF micro_batch_size must be absent or a positive integer, never None")
    if config.do_sample:
        validation=getattr(config,"val_kwargs",None)
        if validation is None or any(not hasattr(validation,key) for key in ("top_k","top_p","temperature")):
            raise ValueError("HF sampled validation requires val_kwargs top_k/top_p/temperature")
    return {"declared_mode":"async","execution":"direct_local_HF_generate_sequences_no_scheduler",
            "micro_batch_size":"use_request_batch_size" if micro_batch is missing else micro_batch,
            "response_length":config.response_length,"do_sample":config.do_sample}


class _OwnedHFLifecycle:
    def _initialize_owned(self, module, device, max_state_bytes):
        import torch
        if not isinstance(module, torch.nn.Module):
            raise TypeError("An owned torch.nn.Module is required")
        if any(type(child).__name__ in ("FullyShardedDataParallel", "DistributedDataParallel")
               for child in module.modules()):
            raise ValueError("This local HF adapter does not support distributed model wrappers")
        target=torch.device(device)
        if target.type not in ("cpu", "musa", "cuda"):
            raise ValueError("Supported explicit devices are cpu, musa and cuda")
        if type(max_state_bytes) is not int or not 0 < max_state_bytes <= 256*1024**2:
            raise ValueError("max_state_bytes must be 1..256 MiB")
        generation=getattr(module,"generation_config",None)
        if generation is not None and getattr(generation,"cache_implementation",None) not in (None,"dynamic"):
            raise ValueError("Only per-generation dynamic HF caches are supported")
        if hasattr(module,"_cache"):
            raise ValueError("A pre-existing persistent HF cache is not supported")
        if not module.state_dict(): raise ValueError("The owned model has no state tensors")
        self.module=module
        self.target_device=target
        self.max_state_bytes=max_state_bytes
        self.lifecycle_state="active"
        self.lifecycle_events=[]
        self._generating=False
        self._verify_device(target)
        self._snapshot_state()  # Validate dense, bounded full state before any mutation.

    def _verify_device(self, expected):
        tensors=list(self.module.parameters())+list(self.module.buffers())
        if any(t.device.type != expected.type or
               (expected.index is not None and t.device.index != expected.index) for t in tensors):
            raise RuntimeError("Owned model tensors are not all on the requested device")

    def _snapshot_state(self):
        import torch
        state=self.module.state_dict(); total=0; result={}
        for name,value in state.items():
            if type(value) is not torch.Tensor or value.layout != torch.strided or value.is_quantized:
                raise ValueError("Only ordinary dense, unquantized full model state is supported")
            total+=value.numel()*value.element_size()
            if total>self.max_state_bytes: raise ValueError("Owned state exceeds max_state_bytes")
            result[name]=value.detach().cpu().clone()
        return result

    def _record(self, action, **fields):
        event={"action":action,"state":self.lifecycle_state,
               "cache_scope":"per_generation_dynamic_no_persistent_cache",**fields}
        self.lifecycle_events.append(event)
        return event

    def _require_idle(self):
        if self._generating: raise RuntimeError("Cannot change lifecycle during generation")
        if self.lifecycle_state=="error": raise RuntimeError("Owned rollout lifecycle is in error state")

    async def release(self):
        """Synchronize and offload this rollout's weights/buffers to CPU."""
        import torch
        self._require_idle()
        before=self._snapshot_state()
        if self.target_device.type != "cpu" and self.lifecycle_state=="active":
            getattr(torch,self.target_device.type).synchronize(self.target_device)
        try:
            self.module.to("cpu")
            self._verify_device(torch.device("cpu"))
            after=self._snapshot_state()
            if any(not torch.equal(before[name],after[name]) for name in before):
                raise RuntimeError("Weights changed while releasing owned rollout")
            self.lifecycle_state="released"
            return self._record("release",weights_verified=True,device="cpu")
        except BaseException:
            self.lifecycle_state="error";raise

    async def resume(self, tags):
        """Restore owned weights; dynamic KV cache is allocated by each request."""
        import torch
        self._require_idle()
        if not isinstance(tags,list) or not tags or any(t not in ("weights","kv_cache") for t in tags):
            raise ValueError("resume tags must contain weights and/or kv_cache")
        if "weights" not in tags and self.lifecycle_state!="active":
            raise RuntimeError("Resume weights before requesting a dynamic KV cache")
        before=self._snapshot_state()
        try:
            if "weights" in tags: self.module.to(self.target_device)
            self._verify_device(self.target_device)
            after=self._snapshot_state()
            if any(not torch.equal(before[name],after[name]) for name in before):
                raise RuntimeError("Weights changed while resuming owned rollout")
            self.lifecycle_state="active"
            return self._record("resume",tags=list(tags),weights_verified=True,device=str(self.target_device))
        except BaseException:
            self.lifecycle_state="error";raise

    async def update_weights(self, weights, **kwargs):
        """Load one complete state, after validating every name, shape and dtype.

        Input snapshots and rollback state are bounded individually by the
        configured state budget. No partial names or implicit casts are accepted.
        """
        import torch
        self._require_idle()
        if kwargs: raise ValueError("Unsupported update_weights options")
        current=self.module.state_dict(); incoming={}; total=0
        for item in weights:
            if not isinstance(item,(tuple,list)) or len(item)!=2:
                raise ValueError("Expected (name, tensor) weight pairs")
            name,value=item
            if name not in current or name in incoming:
                raise ValueError("Unknown or duplicate weight name")
            target=current[name]
            if (type(value) not in (torch.Tensor,torch.nn.Parameter) or value.layout!=torch.strided or
                    value.is_quantized or value.shape!=target.shape or value.dtype!=target.dtype):
                raise ValueError(f"Weight shape/dtype/layout mismatch: {name}")
            total+=value.numel()*value.element_size()
            if total>self.max_state_bytes: raise ValueError("Incoming state exceeds max_state_bytes")
            incoming[name]=value.detach().cpu().clone()
        if incoming.keys()!=current.keys(): raise ValueError("A complete model state is required")
        previous=self._snapshot_state()
        parameter_ids={name:id(value) for name,value in self.module.named_parameters()}
        try:
            self.module.load_state_dict(incoming,strict=True,assign=False)
            after=self._snapshot_state()
            if any(not torch.equal(incoming[name],after[name]) for name in incoming):
                raise RuntimeError("Loaded weight verification failed")
            if parameter_ids!={name:id(value) for name,value in self.module.named_parameters()}:
                raise RuntimeError("Weight update replaced owned Parameter objects")
        except BaseException:
            try: self.module.load_state_dict(previous,strict=True,assign=False)
            except BaseException: self.lifecycle_state="error"
            raise
        return self._record("update_weights",tensor_count=len(incoming),bytes=total,weights_verified=True)

    def generate_sequences(self, prompts):
        self._require_idle()
        if self.lifecycle_state!="active": raise RuntimeError("Resume owned weights before generation")
        self._verify_device(self.target_device)
        previous_training=self.module.training
        self._generating=True
        try:
            return super().generate_sequences(prompts)
        finally:
            self.module.train(previous_training)
            self._generating=False


def create_local_hf_rollout(owned_module, rollout_config, model_config, *, device, max_state_bytes=64*1024**2):
    """Return a concrete subclass using VeRL's actual HF generation implementation.

    ``rollout_config`` and ``model_config`` are the installed VeRL RolloutConfig
    and HFModelConfig objects. The caller loads a separate, unsharded HF model.
    This explicitly repairs the legacy HFRollout constructor's incompatibility
    with the newer BaseRollout constructor, without bypassing abstract checks.
    """
    config_evidence=_validate_hf_config(rollout_config)
    from verl.workers.rollout.hf_rollout import HFRollout
    from verl.workers.rollout.base import BaseRollout

    class LocalHFRollout(_OwnedHFLifecycle,HFRollout):
        def __init__(self):
            BaseRollout.__init__(self,rollout_config,model_config,None)
            self.hf_config_evidence=config_evidence
            self._initialize_owned(owned_module,device,max_state_bytes)

    return LocalHFRollout()
