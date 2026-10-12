"""CPU checks of the VeRL example's acceptance gates, not VeRL execution."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
try:
    from tensordict import TensorDict
except ImportError:
    TensorDict=None


spec=importlib.util.spec_from_file_location("verl_tiny_check",Path(__file__).resolve().parents[1]/"examples"/"verl_tiny_step.py")
example=importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class VerlAcceptanceGateTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng=example.snapshot_rng("cpu")
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        example.restore_rng(self.saved_rng)
        self.temp.cleanup()

    def make_capture(self):
        rank=self.root/"capture"/"rank_00000";rank.mkdir(parents=True)
        events=[];total=0
        for index,kind in enumerate(("input","output","grad_input","grad_output")):
            path=rank/f"tensor_{index}.npy";np.save(path,np.ones((2,3),dtype=np.float32),allow_pickle=False)
            total+=path.stat().st_size
            events.append({"event_id":index,"state":"tensor","module":"model.layers.0.mlp.down_proj",
                           "device":"musa:0","file":path.name,"bytes":path.stat().st_size,
                           "shape":[2,3],"stored_dtype":"float32","kind":kind,
                           "phase":"backward" if kind.startswith("grad_") else "forward",
                           "step":0,"microbatch":0,"call_index":0,
                           "tensor_path":"args[0]" if kind.endswith("input") else "output"})
        (rank/"manifest.json").write_text(json.dumps({"status":"complete","rank":0,"world_size":1,
                                                     "events_count":4,"tensor_bytes":total}))
        (rank/"events.jsonl").write_text("".join(json.dumps(item)+"\n" for item in events))
        return rank,events

    def test_capture_requires_real_finite_forward_and_linked_nonzero_gradients(self):
        rank,events=self.make_capture()
        self.assertEqual(example.validate_actor_capture(rank.parent,"musa")["nonzero_gradient_elements"],12)
        events[2]["call_index"]=99
        (rank/"events.jsonl").write_text("".join(json.dumps(item)+"\n" for item in events))
        with self.assertRaisesRegex(RuntimeError,"not linked"):
            example.validate_actor_capture(rank.parent,"musa")

    def test_capture_rejects_missing_gradient_file_and_nonfinite_values(self):
        rank,_=self.make_capture();path=rank/"tensor_3.npy"
        np.save(path,np.full((2,3),np.nan,dtype=np.float32),allow_pickle=False)
        with self.assertRaisesRegex(RuntimeError,"finite-value"):
            example.validate_actor_capture(rank.parent,"musa")
        path.unlink()
        with self.assertRaisesRegex(RuntimeError,"missing"):
            example.validate_actor_capture(rank.parent,"musa")

    def test_rng_roundtrip_and_independent_saved_states(self):
        before=example.snapshot_rng("cpu")
        expected=(random.random(),np.random.random(),torch.rand(3))
        after=example.snapshot_rng("cpu")
        example.restore_rng(before)
        actual=(random.random(),np.random.random(),torch.rand(3))
        self.assertTrue(example._equal_tree(expected,actual))
        self.assertTrue(example._equal_tree(after,example.snapshot_rng("cpu")))
        self.assertFalse(example._equal_tree(before,after))
        example.save_rng(self.root/"rng",before)
        saved=json.loads((self.root/"rng"/"state.json").read_text())
        self.assertEqual(saved["device"],"cpu")
        np.testing.assert_array_equal(np.load(self.root/"rng"/"torch_cpu.npy",allow_pickle=False),before["torch_cpu"].numpy())
        np.testing.assert_array_equal(np.load(self.root/"rng"/"numpy_keys.npy",allow_pickle=False),before["numpy"][1])

    @unittest.skipIf(TensorDict is None,"Optional VeRL TensorDict dependency is not installed")
    def test_dataproto_copy_rejects_tensor_aliasing_and_preserves_actual_fields(self):
        batch=SimpleNamespace(batch=TensorDict({"old_log_probs":torch.tensor([1.,2.]),"responses":torch.tensor([4,5])},batch_size=[2]),
                              meta_info={"temperature":1.0},non_tensor_batch={"ids":np.array(["a","b"])})
        cloned=example.clone_ppo_batch(batch)
        self.assertEqual(cloned.batch.batch_size,batch.batch.batch_size)
        self.assertEqual(cloned.batch.device,batch.batch.device)
        self.assertEqual(type(cloned),type(batch))
        cloned.batch["responses"].add_(1)
        self.assertTrue(torch.equal(batch.batch["responses"],torch.tensor([4,5])))
        with mock.patch.object(torch.Tensor,"clone",lambda value:value):
            with self.assertRaisesRegex(RuntimeError,"shares storage"):
                example.clone_ppo_batch(batch)

    @unittest.skipIf(TensorDict is None,"Optional VeRL TensorDict dependency is not installed")
    def test_numeric_clone_avoids_serialization_and_rejects_changed_device(self):
        class ProtocolFixture(SimpleNamespace):
            def __getstate__(self):raise AssertionError("DataProto serialization must not run")
        batch=ProtocolFixture(batch=TensorDict({"responses":torch.tensor([4,5])},batch_size=[2]),
                              meta_info={},non_tensor_batch={})
        self.assertTrue(torch.equal(example.clone_ppo_batch(batch).batch["responses"],batch.batch["responses"]))
        # A broken clone implementation changing devices must fail before equal
        # attempts a cross-device operation; meta tensors require no GPU.
        with mock.patch.object(torch.Tensor,"clone",lambda value:torch.empty_like(value,device="meta")):
            with self.assertRaisesRegex(RuntimeError,"changed or shares storage"):
                example.clone_ppo_batch(batch)
        batch.batch=TensorDict({"responses":torch.ones(2,3)},batch_size=[2,3])
        with self.assertRaisesRegex(TypeError,"one batch dimension"):example.clone_ppo_batch(batch)

    def test_baseline_rejects_model_and_parameter_hooks(self):
        model=torch.nn.Linear(2,2)
        handle=model.register_forward_hook(lambda *_:None)
        with self.assertRaisesRegex(RuntimeError,"module has hooks"):example.assert_unhooked_model(model)
        handle.remove()
        handle=model.weight.register_hook(lambda grad:grad)
        with self.assertRaisesRegex(RuntimeError,"parameter has hooks"):example.assert_unhooked_model(model)
        handle.remove();example.assert_unhooked_model(model)

    def test_paired_cpu_update_exact_and_detects_parameter_or_rng_drift(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.25))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        baseline,baseline_optimizer=copy.deepcopy((model,optimizer))
        self.assertEqual({id(p) for p in baseline.parameters()},
                         {id(p) for group in baseline_optimizer.param_groups for p in group["params"]})
        inputs=torch.ones(2,3)
        before=example.snapshot_rng("cpu")
        model(inputs).square().mean().backward();optimizer.step()
        captured_after=example.snapshot_rng("cpu")
        try:
            example.restore_rng(before)
            baseline(inputs.clone()).square().mean().backward();baseline_optimizer.step()
            baseline_after=example.snapshot_rng("cpu")
        finally:example.restore_rng(captured_after)
        self.assertTrue(example._equal_tree(captured_after,baseline_after))
        self.assertTrue(example.compare_ppo_parameters(model,baseline,self.root/"exact")["all_exact"])
        with torch.no_grad():next(baseline.parameters()).flatten()[0].add_(.01)
        self.assertFalse(example.compare_ppo_parameters(model,baseline,self.root/"drift")["all_exact"])
        torch.rand(1)
        self.assertFalse(example._equal_tree(captured_after,example.snapshot_rng("cpu")))
        with torch.no_grad():next(baseline.parameters()).flatten()[0].fill_(float("nan"))
        self.assertFalse(example.compare_ppo_parameters(model,baseline,self.root/"nan")["all_finite"])


if __name__=="__main__":unittest.main()
