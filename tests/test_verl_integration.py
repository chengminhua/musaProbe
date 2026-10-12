from abc import ABC,abstractmethod
import asyncio
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
import unittest
from unittest import mock

import torch
from musaprobe.integrations.verl import create_local_hf_rollout


class HFConfig(SimpleNamespace):
    def get(self,key,default=None): return getattr(self,key,default)


def config_fixture(**overrides):
    return HFConfig(**{"name":"hf","mode":"async","do_sample":False,
                       "temperature":1.0,"response_length":4,"top_p":1.0,"top_k":0,**overrides})


@contextmanager
def fake_verl_contract():
    class BaseRollout(ABC):
        def __init__(self,config,model_config,device_mesh):
            self.config=config;self.model_config=model_config;self.device_mesh=device_mesh
        @abstractmethod
        async def release(self): raise NotImplementedError
        @abstractmethod
        async def resume(self,tags): raise NotImplementedError
        @abstractmethod
        async def update_weights(self,weights,**kwargs): raise NotImplementedError

    class HFRollout(BaseRollout):
        def __init__(self,*args,**kwargs):
            raise AssertionError("The incompatible legacy constructor must not run")
        def generate_sequences(self,inputs):
            self.module.eval()
            if inputs is None:raise RuntimeError("generation failed")
            return self.module(inputs)

    modules={name:ModuleType(name) for name in
             ("verl","verl.workers","verl.workers.rollout","verl.workers.rollout.base","verl.workers.rollout.hf_rollout")}
    modules["verl.workers.rollout.base"].BaseRollout=BaseRollout
    modules["verl.workers.rollout.hf_rollout"].HFRollout=HFRollout
    with mock.patch.dict("sys.modules",modules):yield HFRollout


class VerlLifecycleTests(unittest.TestCase):
    def make(self,**kwargs):
        return create_local_hf_rollout(torch.nn.Linear(2,2),config_fixture(),
                                      SimpleNamespace(path="fixture"),device="cpu",**kwargs)

    def test_concrete_contract_and_real_cpu_weight_refresh(self):
        with fake_verl_contract() as parent:
            rollout=self.make()
            self.assertIsInstance(rollout,parent)
            self.assertEqual(rollout.model_config.path,"fixture")
            self.assertIsNone(rollout.device_mesh)
            self.assertEqual(rollout.hf_config_evidence["micro_batch_size"],"use_request_batch_size")
            original={n:p.clone() for n,p in rollout.module.state_dict().items()}
            updated={n:p+1 for n,p in original.items()}
            ids=[id(p) for p in rollout.module.parameters()]
            asyncio.run(rollout.update_weights(iter(updated.items())))
            self.assertEqual(ids,[id(p) for p in rollout.module.parameters()])
            for name,value in rollout.module.state_dict().items():self.assertTrue(torch.equal(value,updated[name]))
            asyncio.run(rollout.release())
            self.assertEqual(rollout.lifecycle_state,"released")
            with self.assertRaises(RuntimeError):rollout.generate_sequences(torch.ones(1,2))
            asyncio.run(rollout.resume(["weights","kv_cache"]))
            self.assertEqual(rollout.lifecycle_state,"active")
            self.assertEqual(tuple(rollout.generate_sequences(torch.ones(1,2)).shape),(1,2))
            self.assertTrue(all(e["weights_verified"] for e in rollout.lifecycle_events))

    def test_invalid_full_updates_are_rejected_without_mutation(self):
        with fake_verl_contract():
            rollout=self.make();original=rollout._snapshot_state()
            bad=[[("weight",original["weight"])],
                 [("unknown",original["weight"])],
                 [("weight",original["weight"]),("weight",original["weight"])],
                 [("weight",original["weight"].double()),("bias",original["bias"])],
                 [("weight",torch.zeros(3,2)),("bias",original["bias"])]]
            for values in bad:
                with self.subTest(values=[n for n,_ in values]):
                    with self.assertRaises(ValueError):asyncio.run(rollout.update_weights(iter(values)))
                    for n,p in rollout.module.state_dict().items():self.assertTrue(torch.equal(original[n],p))

    def test_generation_failure_restores_mode_and_releases_busy_guard(self):
        with fake_verl_contract():
            rollout=self.make();rollout.module.train()
            with self.assertRaises(RuntimeError):rollout.generate_sequences(None)
            self.assertTrue(rollout.module.training)
            self.assertFalse(rollout._generating)
            rollout._generating=True
            with self.assertRaises(RuntimeError):asyncio.run(rollout.release())
            with self.assertRaises(RuntimeError):asyncio.run(rollout.resume(["weights"]))
            with self.assertRaises(RuntimeError):asyncio.run(rollout.update_weights(iter([])))

    def test_failed_load_rolls_back_original_weights(self):
        with fake_verl_contract():
            rollout=self.make();original=rollout._snapshot_state();real=rollout.module.load_state_dict
            calls=[]
            def fail_after_partial(state,**kwargs):
                calls.append(1)
                if len(calls)==1:
                    with torch.no_grad():rollout.module.weight.copy_(state["weight"])
                    raise RuntimeError("partial load failure")
                return real(state,**kwargs)
            with mock.patch.object(rollout.module,"load_state_dict",side_effect=fail_after_partial):
                with self.assertRaises(RuntimeError):
                    asyncio.run(rollout.update_weights(iter({n:p+1 for n,p in original.items()}.items())))
            self.assertEqual(len(calls),2)
            for n,p in rollout.module.state_dict().items():self.assertTrue(torch.equal(original[n],p))

    def test_budget_and_persistent_cache_are_explicitly_rejected(self):
        with fake_verl_contract():
            with self.assertRaises(ValueError):self.make(max_state_bytes=1)
            module=torch.nn.Linear(2,2);module.generation_config=SimpleNamespace(cache_implementation="static")
            with self.assertRaises(ValueError):
                create_local_hf_rollout(module,config_fixture(),object(),device="cpu")

    def test_legacy_config_incompatibilities_fail_before_model_changes(self):
        with fake_verl_contract():
            module=torch.nn.Linear(2,2)
            for bad in ({"mode":"sync"},{"micro_batch_size":None},{"micro_batch_size":0},
                        {"response_length":None},{"do_sample":None}):
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    create_local_hf_rollout(module,config_fixture(**bad),object(),device="cpu")
            accepted=create_local_hf_rollout(module,config_fixture(micro_batch_size=2),object(),device="cpu")
            self.assertEqual(accepted.hf_config_evidence["micro_batch_size"],2)


if __name__=="__main__":unittest.main()
