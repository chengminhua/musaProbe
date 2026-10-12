"""Explicit bounded advanced probes. All output/cache paths are caller supplied.

Examples (inside an isolated environment with /lab backed by /data)::
  python advanced_probe.py precheck --op all --device musa --output /lab/outputs/precheck
  python advanced_probe.py fp8 --flavor te-block --device musa --output /lab/outputs/fp8
  python advanced_probe.py graph --device musa --output /lab/outputs/graph
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def fp8_probe(output, device, flavor):
    import torch
    from musaprobe.native_fp8 import export_fp8_bundle,decode_fp8_bundle
    if device=="musa": import torch_musa  # noqa: F401
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    result={"status":"unsupported","device":device,"flavor":flavor,"scope":"raw_fp8_serialization_and_cpu_dequantization"}
    try:
        source=torch.linspace(-2,2,1024,dtype=torch.float32).reshape(8,128).to(device)
        if flavor=="native":
            quantized=source.to(torch.float8_e4m3fn)
            expected=quantized.float().cpu()
        else:
            if device!="musa": raise ValueError("MT-TE runtime probe requires MUSA")
            import transformer_engine.pytorch  # noqa: F401
            import transformer_engine_torch as tex
            if flavor=="te-tensor":
                from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
                quantizer=Float8Quantizer(torch.ones(1,device=device),torch.zeros(1,device=device),tex.DType.kFloat8E4M3)
            else:
                from transformer_engine.musa.pytorch.tensor.mtfp8_tensor import MTFP8Quantizer
                quantizer=MTFP8Quantizer(tex.DType.kFloat8E4M3,1,128)
            quantized=quantizer.quantize(source)
            expected=quantized.dequantize(dtype=torch.float32).cpu()
        export_fp8_bundle(quantized,root/"bundle",metadata={"flavor":flavor,"device":device})
        decoded=decode_fp8_bundle(root/"bundle")
        result.update(status="pass" if torch.equal(decoded,expected) else "mismatch",
                      max_abs_error=float((decoded-expected).abs().max()),
                      quantization_max_abs_error=float((expected-source.cpu()).abs().max()))
    except Exception as exc:
        result.update(status="unsupported",reason=f"{type(exc).__name__}: {exc}")
    (root/"fp8_check.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def make_tiny_model(output):
    """Create random local Llama weights and a tiny tokenizer; no downloads."""
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    vocab={"[UNK]":0,"[BOS]":1,"[EOS]":2,"[PAD]":3,**{f"token{i}":i for i in range(4,32)}}
    tokenizer=Tokenizer(WordLevel(vocab,unk_token="[UNK]")); tokenizer.pre_tokenizer=Whitespace()
    wrapped=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token="[UNK]",bos_token="[BOS]",eos_token="[EOS]",pad_token="[PAD]")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        # The target MUSA attention backend accepts a 64-wide head; the former
        # 16-wide toy head could not exercise its real vLLM inference path.
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=256,
                    num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
                    bos_token_id=1,eos_token_id=2,pad_token_id=3))
    # This is a plain, unwrapped CPU model. Save directly so Accelerate does not
    # import optional DeepSpeed/Triton GPU integrations while creating test data.
    from safetensors.torch import save_file
    model.config.architectures=["LlamaForCausalLM"]
    model.config.save_pretrained(root)
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(root/"model.safetensors"),metadata={"format":"pt"})
    wrapped.save_pretrained(root)
    return {"status":"complete","model_dir":str(root),"weights":"random_synthetic","downloaded":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["precheck","fp8","compile","graph","kernel","kernel-inspect","framework","framework-inspect","make-tiny-model"])
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=["cpu","musa","cuda"],default="cpu")
    parser.add_argument("--op",default="aten.linear.default")
    parser.add_argument("--dtype",default="float32",choices=["float16","bfloat16","float32","float64"])
    parser.add_argument("--flavor",choices=["native","te-tensor","te-block"],default="native")
    parser.add_argument("--backend",default=None)
    parser.add_argument("--framework",default="vllm")
    parser.add_argument("--model")
    parser.add_argument("--include-dir")
    args=parser.parse_args()
    # Offline-only HF lookup and bounded compiler caches under the explicit output parent.
    root=Path(args.output).expanduser().resolve()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    for key,leaf in [("XDG_CACHE_HOME","cache"),("TORCHINDUCTOR_CACHE_DIR","inductor"),("TRITON_CACHE_DIR","triton"),("HF_HOME","huggingface")]:
        os.environ.setdefault(key,str(root.parent/(root.name+"_"+leaf)))
    if args.action=="precheck":
        from musaprobe.precheck import run_precheck
        from musaprobe.replay import PURE_OPS
        if args.op=="all":
            root.mkdir(parents=True,exist_ok=False)
            results=[run_precheck(op,root/str(index),device=args.device,dtype=args.dtype) for index,op in enumerate(PURE_OPS)]
            result={"status":"pass" if all(x["status"]=="pass" for x in results) else "incomplete", "cases":results}
            (root/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
        else: result=run_precheck(args.op,root,device=args.device,dtype=args.dtype)
    elif args.action=="fp8": result=fp8_probe(root,args.device,args.flavor)
    elif args.action in ("compile","graph"):
        import torch
        from musaprobe.compiled import validate_execution
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.GELU())
        result=validate_execution(model,(torch.arange(8,dtype=torch.float32).reshape(2,4),),root,
                                  device=args.device,mode=args.action,backend=args.backend)
    elif args.action=="kernel":
        from musaprobe.kernel_debug import run_debug_kernel
        result=run_debug_kernel(root,backend=args.backend or "musa",device=args.device)
    elif args.action=="kernel-inspect":
        from musaprobe.kernel_debug import inspect_kernel_interfaces
        if not args.include_dir: parser.error("kernel-inspect requires --include-dir")
        result=inspect_kernel_interfaces(args.include_dir)
        root.mkdir(parents=True,exist_ok=False)
        (root/"interfaces.json").write_text(json.dumps(result,indent=2)+"\n")
    elif args.action=="framework":
        from musaprobe.frameworks import run_framework_smoke
        result=run_framework_smoke(args.framework,args.model,root,device=args.device)
    elif args.action=="framework-inspect":
        from musaprobe.frameworks import inspect_framework,FRAMEWORKS
        root.mkdir(parents=True,exist_ok=False)
        result={"frameworks":[inspect_framework(name) for name in FRAMEWORKS]}
        (root/"frameworks.json").write_text(json.dumps(result,indent=2)+"\n")
    else: result=make_tiny_model(root)
    print(json.dumps(result,indent=2))
    return 0 if result.get("status") in (None,"pass","complete") else 2


if __name__=="__main__": raise SystemExit(main())
