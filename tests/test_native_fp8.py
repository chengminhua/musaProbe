from pathlib import Path
import tempfile
import unittest
import os
import numpy as np
import torch
from musaprobe.native_fp8 import export_fp8_bundle,load_fp8_bundle,decode_fp8_bundle,fp8_snapshot,_decode_arrays


class FP8Tests(unittest.TestCase):
    def test_native_bits_and_values_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            for dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
                value=torch.tensor([0.,-1.,1.5,32.]).to(dtype)
                target=Path(temp)/str(dtype)
                export_fp8_bundle(value,target)
                bundle=load_fp8_bundle(target)
                np.testing.assert_array_equal(bundle["arrays"]["payload"],value.view(torch.uint8).numpy())
                self.assertTrue(torch.equal(decode_fp8_bundle(target),value.float()))

    def test_regular_float_is_not_claimed_native_fp8(self):
        with self.assertRaises(TypeError): fp8_snapshot(torch.ones(4))

    def test_tampered_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"fp8"
            export_fp8_bundle(torch.ones(3).to(torch.float8_e4m3fn),target)
            np.save(target/"array_0.npy",np.zeros(3,dtype=np.uint8))
            with self.assertRaises(ValueError): load_fp8_bundle(target)

    def test_blockwise_scale_indexing_is_explicit(self):
        bits=torch.ones(3,5).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        info={"format":"torch.float8_e4m3fn","scale_mode":"mt_blockwise","block_m":2,"block_n":3}
        value=_decode_arrays(info,{"payload":bits,"scale_inv":np.array([[2.,3.],[4.,5.]],np.float32)})
        np.testing.assert_array_equal(value.numpy(),[[2,2,2,3,3],[2,2,2,3,3],[4,4,4,5,5]])
        with self.assertRaises(ValueError): _decode_arrays(info,{"payload":bits,"scale_inv":np.ones((3,3),np.float32)})

    def test_per_tensor_scale_roundtrip(self):
        value=torch.tensor([1.,2.,-3.]).to(torch.float8_e4m3fn)
        def adapter(x):
            s=fp8_snapshot(x); s["metadata"]["scale_mode"]="per_tensor"
            s["auxiliary"]={"scale_inv":np.array([0.25],np.float32)}
            return s
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"scaled"
            export_fp8_bundle(value,target,adapter=adapter)
            self.assertTrue(torch.equal(decode_fp8_bundle(target),value.float()*0.25))

    @unittest.skipUnless(os.environ.get("MUSAPROBE_TEST_TE_RUNTIME")=="1",
                         "MT-TE package import requires device runtime; opt in during explicit GPU validation")
    def test_real_te_objects_with_cpu_buffers_when_available(self):
        try:
            import transformer_engine.pytorch  # Registers the packaged native extension.
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor,Float8Quantizer
            from transformer_engine.musa.pytorch.tensor.mtfp8_tensor import MTFP8Tensor,MTFP8Quantizer
        except ImportError as exc:
            self.skipTest(f"Optional installed MT-TE API unavailable: {exc}")
        bits=torch.ones(2,4).to(torch.float8_e4m3fn).view(torch.uint8)
        q=Float8Quantizer(torch.tensor([2.]),torch.tensor([3.]),tex.DType.kFloat8E4M3)
        value=Float8Tensor(bits.shape,dtype=torch.float32,data=bits,fp8_scale_inv=torch.tensor([0.5]),fp8_dtype=tex.DType.kFloat8E4M3,quantizer=q)
        snap=fp8_snapshot(value)
        self.assertEqual(snap["metadata"]["scale_mode"],"per_tensor")
        self.assertTrue(torch.equal(snap["analysis_tensor"],torch.full((2,4),0.5)))
        self.assertEqual(snap["auxiliary"]["quantizer_amax"].tolist(),[3.])
        bq=MTFP8Quantizer(tex.DType.kFloat8E4M3,1,2)
        block=MTFP8Tensor(bits.shape,dtype=torch.float32,rowwise_data=bits,rowwise_scale_inv=torch.tensor([[2.,3.],[4.,5.]]),columnwise_data=None,columnwise_scale_inv=None,fp8_dtype=tex.DType.kFloat8E4M3,quantizer=bq)
        snap=fp8_snapshot(block)
        self.assertEqual(snap["metadata"]["scale_mode"],"mt_blockwise")
        np.testing.assert_array_equal(snap["analysis_tensor"].numpy(),[[2,2,3,3],[4,4,5,5]])


if __name__=="__main__": unittest.main()
