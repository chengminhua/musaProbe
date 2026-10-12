import unittest
import torch
from musaprobe.graph_probe import GraphTensorProbe


class GraphProbeTests(unittest.TestCase):
    def test_transparent_autograd_and_stale_detection(self):
        probe = GraphTensorProbe((2, 3))
        value = torch.ones(2, 3, requires_grad=True)
        rng = torch.get_rng_state().clone()
        result = probe(value)
        self.assertIs(value, result)
        result.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        snapshot, generation = probe.snapshot()
        self.assertEqual(generation, 1)
        value.detach().add_(4)
        self.assertTrue(torch.equal(snapshot, torch.ones_like(snapshot)))
        with self.assertRaises(RuntimeError):
            probe.snapshot()
        self.assertEqual(probe.state_dict(), {})

    def test_static_contract_and_budget(self):
        with self.assertRaises(ValueError):
            GraphTensorProbe((100, 100), max_bytes=1)
        probe = GraphTensorProbe((2, 3))
        with self.assertRaises(ValueError):
            probe(torch.ones(3, 2))
