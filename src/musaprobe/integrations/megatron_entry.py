"""torchrun entry: execute an unchanged training script with opt-in observation."""
from __future__ import annotations
import os
from pathlib import Path
import runpy
import sys


def configure_cpu_threads():
    """Apply explicit per-process limits before importing the training stack."""
    limits = {}
    for key in ("MUSAPROBE_INTRAOP_THREADS", "MUSAPROBE_INTEROP_THREADS"):
        if key in os.environ:
            value = int(os.environ[key])
            if not 1 <= value <= 256:
                raise ValueError(f"{key} must be in 1..256")
            limits[key] = value
    if limits:
        import torch
        if "MUSAPROBE_INTRAOP_THREADS" in limits:
            torch.set_num_threads(limits["MUSAPROBE_INTRAOP_THREADS"])
        if "MUSAPROBE_INTEROP_THREADS" in limits:
            torch.set_num_interop_threads(limits["MUSAPROBE_INTEROP_THREADS"])
        print(f"musaProbe CPU threads: intra={torch.get_num_threads()} "
              f"inter={torch.get_num_interop_threads()}", flush=True)


def finish_distributed_run():
    """Drain successful runs identically for baseline and capture profiles.

    Do not import or initialize a device here. Failed runs must not enter a new
    collective because another rank may already have exited. A cleanup failure
    deliberately escapes to torchrun rather than producing a false exit code 0.
    """
    torch = sys.modules.get("torch")
    dist = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return
    synchronize = None
    if "torch_musa" in sys.modules:
        synchronize = torch.musa.synchronize
    elif getattr(torch, "cuda", None) is not None and torch.cuda.is_initialized():
        synchronize = torch.cuda.synchronize
    if synchronize is not None:
        synchronize()
    dist.barrier()
    if synchronize is not None:
        synchronize()
    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: megatron_entry.py /absolute/pretrain_gpt.py [training arguments]")
    script = str(Path(sys.argv[1]).resolve(strict=True))
    sys.argv = [script, *sys.argv[2:]]
    # A torchrun file entry puts this integrations/ directory at sys.path[0].
    # Its megatron.py helper must not shadow the real top-level megatron package.
    # Reproduce the original script's import search root before any MUSA imports.
    sys.path[0] = str(Path(script).parent)
    configure_cpu_threads()
    probe = None
    if os.environ.get("MUSAPROBE_ENABLED", "0") == "1":
        config = os.environ.get("MUSAPROBE_CONFIG")
        if not config:
            raise RuntimeError("MUSAPROBE_ENABLED=1 requires MUSAPROBE_CONFIG")
        # Match the original entry's import order before touching Megatron symbols.
        if os.environ.get("ACCELERATOR_BACKEND", "musa") == "musa":
            import musa_patch  # noqa: F401
        from musaprobe.integrations.megatron import MegatronProbe
        probe = MegatronProbe(config, injection_path=os.environ.get("MUSAPROBE_INJECTION")).install()
    status = "complete"
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status = "error"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        try:
            if probe is not None:
                probe.stop()
            if status == "complete":
                finish_distributed_run()
        except BaseException:
            status = "error"
            raise
        finally:
            if probe is not None:
                probe.close(status=status)


if __name__ == "__main__":
    main()
