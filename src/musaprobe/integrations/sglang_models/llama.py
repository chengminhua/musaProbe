"""Public SGLang external-model entry; delegates weights/math to its Llama."""
import os

from sglang.srt.models.llama import LlamaForCausalLM as _NativeLlamaForCausalLM
from sglang.srt.server_args import get_global_server_args

from musaprobe.integrations.sglang import _WorkerCapture, _BOOTSTRAP_ENV


class LlamaForCausalLM(_NativeLlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.environ.get(_BOOTSTRAP_ENV)
        if not path:
            raise RuntimeError("Explicit musaProbe SGLang session is required")
        self._musaprobe_runtime = _WorkerCapture(self, get_global_server_args(), path)

    def forward(self, *args, **kwargs):
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        return self._musaprobe_runtime.forward(batch,
            lambda: super(LlamaForCausalLM, self).forward(*args, **kwargs))


EntryClass = LlamaForCausalLM
