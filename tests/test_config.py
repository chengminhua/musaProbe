import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from musaprobe.config import ProbeConfig


class ConfigTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config = ProbeConfig(output_dir=Path(tmp))
            self.assertEqual(config.mode, "statistics")
            self.assertEqual(config.modules, ["*"])
            self.assertFalse(config._run_id_explicit)
            self.assertNotIn("_run_id_explicit", config.to_dict())
            target = Path(tmp) / "config.json"
            target.write_text(json.dumps(config.to_dict()))
            restored = ProbeConfig.from_file(target)
            self.assertEqual(restored.to_dict(), config.to_dict())

    def test_bad_fields_rejected(self):
        cases = [{"typo": 1}, {"steps": [True]}, {"steps": [-1]}, {"steps": "all"},
                 {"mode": "unknown"}, {"capture_inputs": 1}, {"max_events": 0},
                 {"stack_depth": -1}, {"max_rank_bytes": -1}, {"rank": -1},
                 {"world_size": 0}, {"rank": 2, "world_size": 2},
                 {"modules": "*"}, {"apis": [1]}, {"run_id": "../escape"},
                 {"run_id": ".."}, {"run_id": ""}, {"run_id": "a\\b"}]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ProbeConfig.from_dict({"output_dir": "/tmp/probe", "run_id": "test", **invalid})
        with self.assertRaises(ValueError):
            ProbeConfig.from_dict({})
        with self.assertRaises(ValueError):
            ProbeConfig.from_dict([])

    def test_shared_distributed_run_id(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "shared run_id"):
                ProbeConfig(output_dir="/tmp/probe", world_size=2)
        with patch.dict(os.environ, {"MUSAPROBE_RUN_ID": "shared"}, clear=True):
            self.assertEqual(ProbeConfig(output_dir="/tmp/probe", world_size=2).run_id, "shared")
            self.assertEqual(ProbeConfig(output_dir="/tmp/probe", run_id="explicit").run_id, "explicit")

    def test_mutable_defaults_are_not_shared(self):
        first = ProbeConfig(output_dir="/tmp/probe")
        second = ProbeConfig(output_dir="/tmp/probe")
        first.modules.append("extra")
        self.assertEqual(second.modules, ["*"])
        values = [1, 1, 2]
        config = ProbeConfig(output_dir="/tmp/probe", steps=values)
        values.append(3)
        self.assertEqual(config.steps, [1, 2])


if __name__ == "__main__":
    unittest.main()
