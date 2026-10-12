import copy
import json
import tempfile
import unittest

import torch

from musaprobe import PrecisionDebugger
from musaprobe.integrations.optimizer import capture_optimizer_state


class OptimizerTests(unittest.TestCase):
    def test_adam_state_observation_does_not_change_state(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            debugger = PrecisionDebugger(config={"output_dir": directory, "run_id": "optimizer", "mode": "tensor"})
            debugger.start(model)
            result = capture_optimizer_state(debugger, model, optimizer)
            debugger.close()
            records = [json.loads(line) for line in (debugger.rank_dir / "events.jsonl").read_text().splitlines()]
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len([r for r in records if r["kind"] == "optimizer_state"]), 6)
        after = optimizer.state_dict()
        for index, states in before["state"].items():
            for key, value in states.items():
                self.assertTrue(torch.equal(value, after["state"][index][key]))

    def test_unmapped_parameter_is_rejected(self):
        model = torch.nn.Linear(3, 2)
        another = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.Adam([another])
        another.sum().backward()
        optimizer.step()
        result = capture_optimizer_state(None, model, optimizer)
        self.assertEqual(result["status"], "unsupported")

    def test_excluded_truncated_and_pending_states_are_not_reported_as_pass(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        cases = [("excluded", {"modules": ["unmatched"]}, "skipped"),
                 ("truncated", {"max_tensor_bytes": 0}, "incomplete"),
                 ("async", {"async_write": True}, "submitted")]
        with tempfile.TemporaryDirectory() as directory:
            for name, settings, status in cases:
                debugger = PrecisionDebugger(config={"output_dir": directory, "run_id": name,
                    "mode": "tensor", "rank": 0, "world_size": 1, **settings}).start(model)
                try:
                    result = capture_optimizer_state(debugger, model, optimizer)
                    self.assertEqual(result["status"], status)
                    if name == "excluded":
                        self.assertFalse(result["captured"])
                        self.assertTrue(result["skipped"])
                finally:
                    debugger.close()


if __name__ == "__main__":
    unittest.main()
