import unittest
import torch
from musaprobe.environment import capture_training_config


class EnvironmentTests(unittest.TestCase):
    def test_rng_and_input_are_unchanged_and_reproducible(self):
        value = torch.arange(5.0)
        state = torch.get_rng_state().clone()
        first = capture_training_config(inputs={"sample": value})
        second = capture_training_config(inputs={"sample": value})
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        changed = capture_training_config(inputs={"sample": value + 1})
        self.assertNotEqual(first["input_state"], changed["input_state"])

    def test_missing_information_remains_unknown(self):
        result = capture_training_config(include_rng=False)
        self.assertIsNone(result["random_state"])
        self.assertIsNone(result["input_state"])
        self.assertIsNone(result["precision"])

    def test_input_budget(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            capture_training_config(inputs={"x": torch.ones(10)}, max_input_bytes=4)
