from pathlib import Path
import tempfile
import unittest
import torch
from musaprobe.compiled import validate_execution


class ExecutionTests(unittest.TestCase):
    def test_actual_compiler_backend_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.GELU())
            state={k:v.clone() for k,v in model.state_dict().items()}
            r=validate_execution(model,(torch.arange(8,dtype=torch.float32).reshape(2,4),),Path(temp)/"compile",backend="aot_eager")
            self.assertEqual(r["status"],"pass",r)
            self.assertGreater(r["compiled_graphs"],0)
            self.assertEqual(len(r["replays"]),2)
            for k,v in state.items(): self.assertTrue(torch.equal(v,model.state_dict()[k]))
            self.assertTrue(model.training)

    def test_graph_on_cpu_is_explicit_unsupported(self):
        with tempfile.TemporaryDirectory() as temp:
            r=validate_execution(torch.nn.Linear(2,2),(torch.ones(1,2),),Path(temp)/"graph",mode="graph")
            self.assertEqual(r["status"],"unsupported")
            self.assertFalse(r["internal_kernel_coverage"])

    def test_torchada_is_not_a_cpu_compiler_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            r=validate_execution(torch.nn.Linear(2,2),(torch.ones(1,2),),Path(temp)/"compile",backend="torchada")
            self.assertEqual(r["status"],"unsupported")
            self.assertIn("explicit MUSA Inductor adapter",r["reason"])
            self.assertEqual(r["compiled_graphs"],0)


if __name__=="__main__": unittest.main()
