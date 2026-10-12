import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from musaprobe.integrations.vllm import MusaProbeWorkerExtension,generate_with_probe,CaptureRPCError


class Worker(MusaProbeWorkerExtension):
    def __init__(self, eager=True):
        self.model_config=SimpleNamespace(enforce_eager=eager)
        self.model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.ReLU())
    def get_model(self): return self.model


class Engine:
    def __init__(self,worker,fail=False): self.worker,self.fail=worker,fail
    def collective_rpc(self,method,timeout,args=(),kwargs=None):
        return [getattr(self.worker,method)(*args,**(kwargs or {}))]
    def generate(self,prompts,sampling_params,use_tqdm=False):
        if self.fail: raise RuntimeError("controlled request failure")
        with torch.no_grad():
            self.worker.model(torch.ones(1,2))
            self.worker.model(torch.ones(1,2)*2)
        return [SimpleNamespace(request_id="native-7")]


class VLLMIntegrationTests(unittest.TestCase):
    def test_actual_module_events_context_and_hook_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            worker=Worker(); engine=Engine(worker)
            result=generate_with_probe(engine,"test",None,config={"output_dir":temp,"run_id":"capture","mode":"tensor","modules":["0"]},request_tag="request-a")
            self.assertTrue(result["cleanup_verified"])
            self.assertEqual(result["workers"][0]["model_forward_calls"],2)
            rank=Path(temp)/"capture/rank_00000"
            manifest=json.loads((rank/"manifest.json").read_text())
            self.assertEqual(manifest["metadata"]["native_request_ids"],["native-7"])
            events=[json.loads(line) for line in (rank/"events.jsonl").read_text().splitlines()]
            self.assertEqual({e["engine_forward_call"] for e in events},{0,1})
            self.assertEqual({e["request_tag"] for e in events},{"request-a"})
            self.assertEqual({e["engine_phase"] for e in events},{"unknown"})
            for module in worker.model.modules():
                self.assertFalse(module._forward_pre_hooks)
                self.assertFalse(module._forward_hooks)
            count=len(events)
            worker.model(torch.ones(1,2))
            self.assertEqual(len((rank/"events.jsonl").read_text().splitlines()),count)

    def test_request_exception_marks_manifest_error_and_removes_hooks(self):
        with tempfile.TemporaryDirectory() as temp:
            worker=Worker()
            with self.assertRaises(CaptureRPCError) as caught:
                generate_with_probe(Engine(worker,fail=True),"test",None,config={"output_dir":temp,"run_id":"capture"},request_tag="failed")
            self.assertTrue(caught.exception.report["cleanup_verified"])
            manifest=json.loads((Path(temp)/"capture/rank_00000/manifest.json").read_text())
            self.assertEqual(manifest["status"],"error")
            self.assertTrue(any("controlled request failure" in e for e in manifest["metadata"]["collector_errors"]))
            self.assertFalse(worker.model._forward_pre_hooks)

    def test_graph_and_dispatch_config_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CaptureRPCError):
                generate_with_probe(Engine(Worker(eager=False)),"test",None,config={"output_dir":temp},request_tag="x")
            with self.assertRaisesRegex(ValueError,"thread-local"):
                generate_with_probe(Engine(Worker()),"test",None,config={"output_dir":temp,"apis":["aten.add.Tensor"]},request_tag="x")

    def test_stop_cannot_remove_another_request_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            worker=Worker()
            self.assertEqual(worker.musaprobe_start({"output_dir":temp},"owner","x")["status"],"started")
            refused=worker.musaprobe_stop("stranger")
            self.assertFalse(refused["hooks_removed"])
            self.assertIsNotNone(worker._musaprobe_capture_state)
            stopped=worker.musaprobe_stop("owner",error="test completion")
            self.assertTrue(stopped["hooks_removed"])

    def test_keyboard_interrupt_still_closes_capture_as_error(self):
        with tempfile.TemporaryDirectory() as temp:
            worker=Worker();engine=Engine(worker)
            def interrupted(*args,**kwargs):raise KeyboardInterrupt()
            engine.generate=interrupted
            with self.assertRaises(KeyboardInterrupt):
                generate_with_probe(engine,"test",None,config={"output_dir":temp,"run_id":"interrupt"},request_tag="x")
            self.assertIsNone(worker._musaprobe_capture_state)
            manifest=json.loads((Path(temp)/"interrupt/rank_00000/manifest.json").read_text())
            self.assertEqual(manifest["status"],"error")

    def test_rpc_cleanup_failure_is_never_reported_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            worker=Worker();engine=Engine(worker)
            original=engine.collective_rpc
            def broken(method,*args,**kwargs):
                if method=="musaprobe_stop":raise TimeoutError("controlled RPC timeout")
                return original(method,*args,**kwargs)
            engine.collective_rpc=broken
            try:
                with self.assertRaises(CaptureRPCError) as caught:
                    generate_with_probe(engine,"test",None,config={"output_dir":temp},request_tag="x")
                self.assertFalse(caught.exception.report["cleanup_verified"])
                self.assertIn("TimeoutError",caught.exception.report["cleanup_error"])
            finally:
                state=worker._musaprobe_capture_state
                if state: worker.musaprobe_stop(state["token"],error="test cleanup")


if __name__=="__main__":unittest.main()
