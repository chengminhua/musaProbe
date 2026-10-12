from pathlib import Path
import tempfile
import unittest
import json
import numpy as np
import torch
from musaprobe.precheck import run_precheck,precheck_captured
from musaprobe.replay import PURE_OPS


class PrecheckTests(unittest.TestCase):
    def test_all_pure_ops_forward_and_vjp(self):
        with tempfile.TemporaryDirectory() as temp:
            for index,op in enumerate(PURE_OPS):
                with self.subTest(op=op):
                    r=run_precheck(op,Path(temp)/str(index))
                    self.assertEqual(r["status"],"pass",r)
                    self.assertEqual(r["provenance"]["kind"],"synthetic")
                    self.assertEqual(r["backward_scope"],"synthetic_vector_jacobian_product")
                    self.assertTrue((Path(r["directory"])/r["files"]["gradient_seed"]).exists())

    def test_seed_is_reproducible_without_changing_global_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            rng=torch.get_rng_state().clone()
            a=run_precheck("aten.linear.default",Path(temp)/"a",seed=42)
            b=run_precheck("aten.linear.default",Path(temp)/"b",seed=42)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name in a["files"]:
                np.testing.assert_array_equal(np.load(Path(a["directory"])/a["files"][name]),np.load(Path(b["directory"])/b["files"][name]))

    def test_unsafe_op_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): run_precheck("os.system",Path(temp)/"bad")
            run_precheck("aten.gelu.default",Path(temp)/"case",backward=False)
            with self.assertRaises(FileExistsError): run_precheck("aten.gelu.default",Path(temp)/"case")

    def test_small_inputs_cannot_expand_into_unbounded_output(self):
        from musaprobe.precheck import _check
        with tempfile.TemporaryDirectory() as temp:
            r=_check("aten.mm.default",(torch.ones(4096,1),torch.ones(1,4096)),{},Path(temp)/"large",
                     device="cpu",backward=False,gradient_seed=0,atol=1e-5,rtol=1e-4,
                     provenance={"kind":"synthetic"})
            self.assertEqual(r["status"],"unsupported")
            self.assertIn("output exceeds",r["reason"])
            self.assertNotIn("reference_output",r["files"])

    def test_captured_input_vjp_does_not_claim_training_backward(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rank=root/"rank"; rank.mkdir()
            spec={"args":{"type":"tuple","items":[{"tensor_path":"$args[0]"},{"tensor_path":"$args[1]"}]},"kwargs":{"type":"dict","items":[]}}
            events=[]
            for i,(array,kind,path) in enumerate([(np.ones(2,np.float32),"api_input","$args[0]"),(np.ones(2,np.float32),"api_input","$args[1]"),(np.ones(2,np.float32)*2,"api_output","$output")]):
                np.save(rank/f"{i}.npy",array)
                events.append({"schema_version":1,"event_id":i,"step":0,"microbatch":0,"phase":"forward","module":"layer","call_index":0,"api_call_index":0,"op":"aten.add.Tensor","kind":kind,"tensor_path":path,"call_spec":spec,"state":"tensor","file":f"{i}.npy","dtype":"torch.float32","shape":[2]})
            (rank/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            r=precheck_captured(rank,2,root/"vjp")
            self.assertEqual(r["status"],"pass",r)
            self.assertEqual(r["provenance"]["kind"],"captured_inputs")
            self.assertFalse(r["provenance"]["original_training_backward_replayed"])


if __name__=="__main__": unittest.main()
