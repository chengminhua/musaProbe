"""CPU contract fixtures; actual SGLang/MUSA acceptance uses the example."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from musaprobe.integrations.sglang import SGLangCaptureSession, SGLangCaptureError, _WorkerCapture


def server_args(**changes):
    values = dict(tp_size=1, pp_size=1, dp_size=1, nnodes=1, max_running_requests=1,
                  disable_cuda_graph=True, disable_overlap_schedule=True, disable_radix_cache=True,
                  enable_torch_compile=False, speculative_algorithm=None, disaggregation_mode="null")
    return SimpleNamespace(**dict(values, **changes))


class FixtureLlama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
        self.quant_config = None
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(4, 4)
        self.model.layers = torch.nn.ModuleList([layer])

    def forward(self, value):
        return self.model.layers[0].mlp.down_proj(value)


class FixtureEngine:
    def __init__(self, session, fail=False):
        self.server_args = server_args()
        self.model = FixtureLlama()
        self.runtime = _WorkerCapture(self.model, self.server_args, session.root / "bootstrap.json")
        self.fail = fail

    def generate(self, input_ids, sampling_params):
        def invoke():
            value = self.model(torch.ones(1, 4))
            if self.fail:
                raise RuntimeError("controlled original worker failure")
            return value
        with torch.no_grad():
            for mode in ("EXTEND", "DECODE"):
                self.runtime.forward(SimpleNamespace(forward_mode=SimpleNamespace(name=mode), batch_size=1), invoke)
        return {"output_ids": [4, 5]}


class SGLangIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        clean = patch.dict(os.environ)
        clean.start()
        self.addCleanup(clean.stop)
        os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
        os.environ.pop("MUSAPROBE_SGLANG_SESSION", None)

    def session(self, **kwargs):
        session = SGLangCaptureSession(self.root / "session", **kwargs)
        self.addCleanup(session.close)
        return session

    def request(self, session, engine):
        return session.generate(engine, input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4, "temperature": 0}, request_tag="request")

    def assert_unhooked(self, model):
        for module in model.modules():
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)

    def test_worker_tensor_files_native_modes_and_per_forward_cleanup(self):
        session = self.session(config={"async_write": True})
        engine = FixtureEngine(session)
        before_rng = torch.get_rng_state().clone()
        before = engine.generate([1], {})
        self.assertFalse(list((session.root / "forwards").iterdir()))
        result = self.request(session, engine)
        after = engine.generate([1], {})
        self.assertEqual(before, result["output"])
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(result["cleanup_verified"])
        self.assertEqual(len(result["forwards"]), 2)
        self.assertEqual({r["native_forward_mode"] for r in result["forwards"]}, {"EXTEND", "DECODE"})
        for row in result["forwards"]:
            rank = Path(row["rank_dir"])
            events = [json.loads(line) for line in (rank / "events.jsonl").read_text().splitlines()]
            self.assertTrue(events)
            self.assertTrue(all(e["state"] == "tensor" and (rank / e["file"]).is_file() for e in events))
        self.assert_unhooked(engine.model)
        self.assertEqual(len(list((session.root / "forwards").iterdir())), 2)
        self.assertFalse(json.loads((session.root / "control.json").read_text())["enabled"])

    def test_original_forward_failure_disarms_and_closes_worker(self):
        session = self.session()
        engine = FixtureEngine(session, fail=True)
        with self.assertRaises(SGLangCaptureError) as caught:
            self.request(session, engine)
        self.assertIn("controlled original worker failure", caught.exception.report["request_error"])
        self.assertTrue(caught.exception.report["cleanup_verified"])
        self.assert_unhooked(engine.model)
        self.assertFalse(json.loads((session.root / "control.json").read_text())["enabled"])
        manifest = json.loads((session.root / "captures/forward_0000/rank_00000/manifest.json").read_text())
        self.assertEqual(manifest["status"], "error")

    def test_missing_worker_or_unknown_configuration_never_passes(self):
        for options in ({"apis": ["aten.add.*"]}, {"capture_gradients": True}, {"max_events": 129}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SGLangCaptureSession(self.root / "bad", config=options)
        session = self.session()
        with self.assertRaises(FileNotFoundError):
            self.request(session, SimpleNamespace(server_args=server_args()))
        self.assertFalse(json.loads((session.root / "control.json").read_text())["enabled"])
        with self.assertRaisesRegex(ValueError, "disable_overlap_schedule"):
            _WorkerCapture(FixtureLlama(), server_args(disable_overlap_schedule=False), session.root / "bootstrap.json")

    def test_tensor_budget_failure_is_not_clean_generation(self):
        session = self.session(config={"max_tensor_bytes": 1})
        engine = FixtureEngine(session)
        with self.assertRaises(SGLangCaptureError) as caught:
            self.request(session, engine)
        self.assertIn("complete tensor payload", caught.exception.report["validation_error"])
        self.assertTrue(caught.exception.report["cleanup_verified"])
        self.assert_unhooked(engine.model)

    def test_keyboard_interrupt_disarms_and_retains_failure(self):
        session = self.session()
        engine = FixtureEngine(session)
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        engine.generate = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.request(session, engine)
        self.assertFalse(json.loads((session.root / "control.json").read_text())["enabled"])
        self.assertEqual(json.loads((session.root / "request.json").read_text())["status"], "error")

    def test_refuses_second_request_and_preserves_unrelated_extension(self):
        session = self.session()
        engine = FixtureEngine(session)
        self.request(session, engine)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.request(session, engine)
        session.close()
        self.assertNotIn("SGLANG_EXTERNAL_MODEL_PACKAGE", os.environ)
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "other_application"
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            SGLangCaptureSession(self.root / "another")
        self.assertEqual(os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"], "other_application")


if __name__ == "__main__":
    unittest.main()
