"""CPU checks for the acceptance script's independent persistence verifier."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "fp8_capture_check", Path(__file__).resolve().parents[1] / "examples/fp8_capture_check.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
verify = _MODULE._verify_saved_state


class FP8PersistenceAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, flavor="per_tensor"):
        raw = np.arange(8, dtype=np.uint8).reshape(2, 4)
        fields = {"data": raw}
        if flavor == "per_tensor":
            fields.update(fp8_scale_inv=np.ones(1, np.float32),
                          quantizer_scale=np.ones(1, np.float32),
                          quantizer_amax=np.array([2], np.float32), data_transpose=raw.T.copy())
        elif flavor == "mt_blockwise":
            fields = {"rowwise_data": raw, "rowwise_scale_inv": np.ones((2, 1), np.float32),
                      "columnwise_data": raw.copy(), "columnwise_scale_inv": np.ones((1, 4), np.float32)}
        names = {"data": "payload", "rowwise_data": "payload", "fp8_scale_inv": "scale_inv",
                 "rowwise_scale_inv": "scale_inv"}
        before, event = {}, {"auxiliary": {}}
        for source_name, array in fields.items():
            before[source_name] = {"dtype": str(array.dtype), "shape": list(array.shape),
                                   "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            stored_name = names.get(source_name, source_name)
            filename = stored_name + ".npy"
            np.save(self.root / filename, array, allow_pickle=False)
            if stored_name == "payload":
                event["file"] = filename
            else:
                event["auxiliary"][stored_name] = filename
        return before, event

    def test_all_three_actual_field_layouts_verify_without_torch(self):
        for flavor in ("native", "per_tensor", "mt_blockwise"):
            with self.subTest(flavor=flavor):
                before, event = self.fixture(flavor)
                result = verify(self.root, event, before)
                self.assertEqual(len(result), len(before))
                self.assertEqual({item["source_field"] for item in result.values()}, set(before))

    def test_bad_amax_and_columnwise_content_fail_even_when_decode_ignores_them(self):
        for flavor, field in (("per_tensor", "quantizer_amax"), ("mt_blockwise", "columnwise_data"),
                              ("mt_blockwise", "columnwise_scale_inv")):
            with self.subTest(field=field):
                before, event = self.fixture(flavor)
                path = self.root / event["auxiliary"][field]
                array = np.load(path, allow_pickle=False)
                array.flat[0] += 1
                np.save(path, array, allow_pickle=False)
                with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                    verify(self.root, event, before)

    def test_missing_amax_and_columnwise_attachments_fail(self):
        for flavor, field in (("per_tensor", "quantizer_amax"), ("mt_blockwise", "columnwise_data"),
                              ("mt_blockwise", "columnwise_scale_inv")):
            with self.subTest(field=field):
                before, event = self.fixture(flavor)
                del event["auxiliary"][field]
                with self.assertRaisesRegex(RuntimeError, "field mismatch.*missing"):
                    verify(self.root, event, before)

    def test_shape_dtype_and_payload_corruption_fail(self):
        for change in (lambda x: x.reshape(-1), lambda x: x.astype(np.int16), lambda x: x + np.uint8(1)):
            before, event = self.fixture("native")
            path = self.root / event["file"]
            np.save(path, change(np.load(path, allow_pickle=False)), allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify(self.root, event, before)

    def test_unknown_or_ambiguous_runtime_fields_and_extra_attachment_fail(self):
        before, event = self.fixture("native")
        for field in ("new_unmapped_data", "rowwise_data"):
            changed = {**before, field: before["data"]}
            with self.assertRaisesRegex(RuntimeError, "Unmapped|Ambiguous"):
                verify(self.root, event, changed)
        event["auxiliary"]["unexpected"] = "unexpected.npy"
        with self.assertRaisesRegex(RuntimeError, "field mismatch.*extra"):
            verify(self.root, event, before)

    def test_unsafe_and_duplicate_file_references_fail(self):
        before, event = self.fixture("per_tensor")
        for reference in ("../payload.npy", str(self.root / "payload.npy")):
            changed = copy.deepcopy(event)
            changed["file"] = reference
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                verify(self.root, changed, before)
        event["auxiliary"]["quantizer_scale"] = event["auxiliary"]["scale_inv"]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            verify(self.root, event, before)

    def test_missing_payload_or_auxiliary_file_fails(self):
        for field in ("payload", "quantizer_amax"):
            before, event = self.fixture("per_tensor")
            filename = event["file"] if field == "payload" else event["auxiliary"][field]
            (self.root / filename).unlink()
            with self.assertRaises(FileNotFoundError):
                verify(self.root, event, before)


if __name__ == "__main__":
    unittest.main()
