import json

import numpy as np

from analysis_fixtures import FixtureCase
from musaprobe.compare import compare_runs
from musaprobe.fp8_numpy import decode_bits, FP8Tensor, FORMATS


class FP8NumpyTests(FixtureCase):
    def test_all_256_bit_patterns_match_torch_for_four_formats(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch oracle unavailable")
        raw = np.arange(256, dtype=np.uint8)
        for format_name in FORMATS:
            expected = torch.from_numpy(raw.copy()).view(getattr(torch, format_name)).float().numpy()
            actual = decode_bits(raw, format_name)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.signbit(actual[np.isfinite(expected)]), np.signbit(expected[np.isfinite(expected)]))

    def fp8(self, name, raw, *, scale=None, mode="per_tensor", info=None):
        metadata = {"format": "torch.float8_e4m3fn", "encoding": "raw_float8_bits", "logical_shape": list(raw.shape),
                    "scale_mode": mode, **(info or {})}
        event = {"fp8": metadata, "dtype": "torch.bfloat16"}
        if scale is not None:
            event["auxiliary"] = {"scale_inv": "scale.npy"}
        root = self.capture(name, [(raw, event)])
        if scale is not None:
            np.save(root / "rank_00000" / "scale.npy", np.asarray(scale, np.float32), allow_pickle=False)
        return root

    def test_scaled_raw_bytes_do_not_masquerade_as_integer_equality(self):
        raw = np.array([56, 64], np.uint8)  # e4m3fn 1 and 2
        a = self.fp8("a", raw, scale=[1])
        b = self.fp8("b", raw, scale=[2])
        result = compare_runs(a, b, atol=0, rtol=0)
        self.assertEqual(result["records"][0]["status"], "mismatch")
        self.assertEqual(result["records"][0]["metrics"]["max_abs_error"], 2)
        self.assertFalse(result["records"][0]["metrics"]["exact_comparison"])
        self.assertEqual(compare_runs(a, a)["overall"], "pass")
        self.assertEqual(compare_runs(a, a, max_ulp=1)["records"][0]["status"], "unsupported")

    def test_blockwise_decoder_and_nonfinite(self):
        raw = np.full((3, 4), 56, np.uint8)
        path = self.fp8("block", raw, scale=[[1, 2], [3, 4]], mode="mt_blockwise", info={"block_m": 2, "block_n": 2})
        event = json.loads((path / "rank_00000/events.jsonl").read_text())
        tensor = FP8Tensor(path / "rank_00000", event)
        np.testing.assert_array_equal(tensor.flat[3:10], [2, 1, 1, 2, 2, 3, 3])
        nan = self.fp8("nan", np.array([127], np.uint8), mode="unscaled_native_dtype")
        self.assertEqual(compare_runs(nan, nan)["records"][0]["status"], "nonfinite")

    def test_fp8_requires_scales_layout_and_safe_files(self):
        raw = np.array([56], np.uint8)
        missing = self.fp8("missing", raw)
        self.assertEqual(compare_runs(missing, missing)["records"][0]["status"], "error")
        badscale = self.fp8("scale", raw, scale=[float("nan")])
        self.assertEqual(compare_runs(badscale, badscale)["records"][0]["status"], "error")
        one = self.fp8("one", raw, scale=[1])
        other = self.fp8("other", raw, scale=[1], info={"format": "torch.float8_e5m2"})
        self.assertEqual(compare_runs(one, other)["records"][0]["status"], "unaligned")
        path = one / "rank_00000/events.jsonl"
        event = json.loads(path.read_text())
        event["auxiliary"]["scale_inv"] = "../outside.npy"
        path.write_text(json.dumps(event) + "\n")
        self.assertIn("unsafe", compare_runs(one, one)["records"][0]["reason"])
