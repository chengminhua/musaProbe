import unittest

from musaprobe.distributed import agree_proposals


class BoundaryTests(unittest.TestCase):
    def proposal(self, **changes):
        return {"step": 2, "config_version": 0, "stop": False, "update": None, **changes}

    def test_same_configuration_and_stop(self):
        result = agree_proposals([self.proposal(update={"rank": 0, "enabled": True}),
                                  self.proposal(update={"rank": 1, "enabled": True}, stop=True)])
        self.assertTrue(result["accepted"])
        self.assertTrue(result["stop"])

    def test_bad_update_keeps_old_configuration(self):
        result = agree_proposals([self.proposal(update={"enabled": True}), self.proposal()])
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["update"])
        self.assertIn("different", result["error"])

    def test_mismatched_boundary_stops(self):
        result = agree_proposals([self.proposal(), self.proposal(step=3)])
        self.assertTrue(result["stop"])
        self.assertFalse(result["accepted"])

    def test_local_validation_error_is_shared(self):
        result = agree_proposals([self.proposal(), self.proposal(error="invalid budget")])
        self.assertIn("rank 1", result["error"])
        self.assertFalse(result["accepted"])

    def test_large_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            agree_proposals([self.proposal(update={"text": "x" * (1024 * 1024)})])


if __name__ == "__main__":
    unittest.main()
