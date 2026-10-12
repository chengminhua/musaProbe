import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import ModuleType,SimpleNamespace
import numpy as np
from musaprobe.frameworks import import_tensor_bundle,inspect_framework,run_framework_smoke


class FrameworkTests(unittest.TestCase):
    def test_owned_vllm_engine_cleanup_on_success_and_failure(self):
        for fail_request,fail_cleanup,expected in ((False,False,"pass"),(True,False,"unsupported"),(False,True,"error")):
            with self.subTest(fail_request=fail_request,fail_cleanup=fail_cleanup),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);source=root/"model";source.mkdir();(source/"config.json").write_text("{}")
                shutdown=mock.Mock(side_effect=RuntimeError("shutdown failed") if fail_cleanup else None)
                engine=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)),
                    generate=mock.Mock(side_effect=RuntimeError("request failed") if fail_request else None,
                                       return_value=[SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4])])]))
                vllm=ModuleType("vllm");vllm.LLM=mock.Mock(return_value=engine);vllm.SamplingParams=lambda **kw:kw
                platforms=ModuleType("vllm.platforms");platforms.current_platform=SimpleNamespace(device_type="musa")
                with mock.patch.dict("sys.modules",{"vllm":vllm,"vllm.platforms":platforms}):
                    result=run_framework_smoke("vllm",source,root/"output")
                self.assertEqual(result["status"],expected)
                shutdown.assert_called_once_with()
                if fail_request:self.assertTrue((root/"output"/"traceback.txt").is_file())

    def test_safe_import_is_not_framework_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/"source"; source.mkdir()
            np.save(source/"x.npy",np.arange(4,dtype=np.float32))
            manifest=source/"bundle.json"
            manifest.write_text(json.dumps({"framework":"mindspore","tensors":[{"name":"layer.output","file":"x.npy","shape":[4],"dtype":"float32"}]}))
            r=import_tensor_bundle(manifest,root/"output")
            self.assertEqual(r["status"],"complete")
            self.assertFalse(r["backend_execution_verified"])
            event=json.loads((Path(r["rank_dir"])/"events.jsonl").read_text())
            self.assertEqual(event["module"],"layer.output")

    def test_escape_and_dtype_mismatch_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/"source"; source.mkdir()
            np.save(root/"outside.npy",np.ones(2,dtype=np.float32))
            manifest=source/"bundle.json"
            manifest.write_text(json.dumps({"framework":"pytorch","tensors":[{"name":"x","file":"../outside.npy"}]}))
            with self.assertRaises(ValueError): import_tensor_bundle(manifest,root/"out")
            self.assertFalse((root/"out").exists())

    def test_discovery_is_never_execution_proof(self):
        r=inspect_framework("pytorch")
        self.assertTrue(r["installed"])
        self.assertFalse(r["backend_execution_verified"])

    def test_imported_bundles_work_with_numeric_comparison(self):
        from musaprobe.compare import compare_runs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); outputs=[]
            for name,values in [("ms",[1.,2.]),("pt",[1.,3.])]:
                source=root/(name+"_source"); source.mkdir()
                np.save(source/"x.npy",np.asarray(values,np.float32))
                manifest=source/"bundle.json"
                manifest.write_text(json.dumps({"framework":"mindspore" if name=="ms" else "pytorch","tensors":[{"name":"semantic.layer.output","file":"x.npy"}]}))
                output=root/(name+"_capture")
                import_tensor_bundle(manifest,output)
                outputs.append(output)
            result=compare_runs(*outputs)
            self.assertEqual(result["overall"],"fail")
            self.assertEqual(result["summary"]["statuses"],{"mismatch":1})


if __name__=="__main__": unittest.main()
