import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from musaprobe.storage import RunWriter


class StorageTests(unittest.TestCase):
    def config(self, root, **kwargs):
        values = dict(output_dir=root, run_id="test", mode="tensor", max_events=100,
                      max_tensor_bytes=65536, max_rank_bytes=1048576)
        values.update(kwargs)
        c = SimpleNamespace(**values)
        c.to_dict = lambda: values
        return c

    def test_bfloat16_nonfinite_empty(self):
        with tempfile.TemporaryDirectory() as root:
            writer = RunWriter(self.config(root))
            value = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
            event = writer.write_tensor(value)
            self.assertEqual(event["state"], "tensor")
            self.assertEqual(event["dtype"], "torch.bfloat16")
            np.testing.assert_array_equal(np.load(writer.rank_dir / event["file"], allow_pickle=False), value.float().numpy())
            nonfinite = writer.write_tensor(torch.tensor([float("nan"),float("inf"),-float("inf")]))
            self.assertEqual(nonfinite["stats"]["finite_count"],0)
            self.assertEqual(nonfinite["stats"]["nan_count"],1)
            self.assertIsNone(nonfinite["stats"]["mean"])
            empty = writer.write_tensor(torch.empty(0))
            self.assertEqual(empty["stats"]["numel"],0)
            writer.close()
            manifest=json.loads((writer.rank_dir/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')

    def test_budgets_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            config=self.config(root,max_tensor_bytes=2,max_events=1)
            writer=RunWriter(config)
            event=writer.write_tensor(torch.ones(4))
            self.assertEqual(event['state'],'truncated')
            self.assertIsNone(writer.write_tensor(torch.ones(1)))
            self.assertIsNone(writer.write_tensor(torch.ones(1)))
            self.assertEqual(writer.events_count,2)
            writer.close()
            with self.assertRaises(FileExistsError):
                RunWriter(config)

    def test_unsupported_layout_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            writer=RunWriter(self.config(root))
            event=writer.write_tensor(torch.ones(2,dtype=torch.complex64))
            self.assertEqual(event['state'],'unsupported')
            writer.close()

    def test_large_finite_statistics_do_not_overflow_intermediate_reductions(self):
        for tensor in (torch.tensor([1e100, 2e100], dtype=torch.float64),
                       torch.tensor([3e38, 3e38], dtype=torch.float32)):
            stats = RunWriter._stats(tensor)
            values = tensor.double().numpy()
            scale = float(np.max(np.abs(values)))
            self.assertEqual(stats['finite_count'], 2)
            self.assertAlmostEqual(stats['min'] / float(values.min()), 1.0, places=6)
            self.assertAlmostEqual(stats['max'] / float(values.max()), 1.0, places=6)
            self.assertAlmostEqual(stats['mean'] / float(values.mean()), 1.0, places=6)
            self.assertAlmostEqual(stats['l2'] / (float(np.linalg.norm(values / scale)) * scale), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
