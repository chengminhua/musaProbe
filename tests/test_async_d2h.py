"""CPU transport simulations plus explicit unsupported checks; no GPU proof."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musaprobe import PrecisionDebugger
from musaprobe.config import ProbeConfig
from musaprobe.storage import RunWriter
from musaprobe.async_transfer import AsyncD2HSubmissionError


class CpuTransportFixture:
    """Simulates only immutable payload/ownership; never used outside tests."""
    def __init__(self, capacity):
        self.slots = threading.BoundedSemaphore(capacity)
        self.active = 0
        self.maximum = 0
        self.released = 0
        self.fail_wait = False
        self.lock = threading.Lock()
        self.abandoned = []

    def abandon(self, job, reason):
        self.abandoned.append(job)

    def drain_abandoned(self):
        for job in self.abandoned:
            if not job.released:
                job.wait()
                job.release()
        self.abandoned.clear()
        return []

    def close(self):
        pass

    def capture(self, tensor):
        if not self.slots.acquire(timeout=3):
            raise RuntimeError("fixture slot timeout")
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        pool = self
        frozen = tensor.detach().clone()
        class Job:
            released = False
            def wait(self):
                if pool.fail_wait:
                    raise RuntimeError("controlled transfer wait failure")
                return frozen
            def release(self):
                if self.released:
                    return
                self.released = True
                with pool.lock:
                    pool.active -= 1
                    pool.released += 1
                pool.slots.release()
        return Job()


class AsyncD2HTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = ProbeConfig(output_dir=self.tmp.name, run_id="capture", mode="tensor",
                                  rank=0, world_size=1, async_write=True, async_d2h=True,
                                  async_queue_size=2, stack_depth=0)

    def records(self, writer):
        writer.flush()
        return [json.loads(line) for line in (writer.rank_dir / "events.jsonl").read_text().splitlines()]

    def test_requires_async_writer_and_cpu_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "requires async_write"):
            ProbeConfig(output_dir=self.tmp.name, async_d2h=True)
        writer = RunWriter(self.config)
        self.addCleanup(writer.close)
        writer.write_tensor(torch.ones(2))
        row, = self.records(writer)
        self.assertEqual(row["state"], "unsupported")
        self.assertIn("plain dense MUSA", row["reason"])
        self.assertIsNone(row["file"])
        self.assertFalse(any(key.startswith("_") for key in row))

    def test_mock_transport_snapshots_are_immutable_and_final_format_is_regular_npy(self):
        with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
            writer = RunWriter(self.config)
        self.addCleanup(writer.close)
        value = torch.zeros(4, dtype=torch.bfloat16)
        for index in range(7):
            value.fill_(index)
            event = writer.write_tensor(value, config_version=3, backward_id="grad", step=2)
            self.assertEqual(event["state"], "pending")
        writer.close()
        records = self.records(writer)
        self.assertEqual(len(records), 7)
        for index, event in enumerate(records):
            self.assertEqual(event["state"], "tensor")
            self.assertFalse(any(key.startswith("_") for key in event))
            self.assertEqual((event["config_version"], event["backward_id"]), (3, "grad"))
            self.assertEqual(event["stats_device"], "cpu")
            self.assertEqual(event["dtype"], "torch.bfloat16")
            self.assertEqual(event["stored_dtype"], "float32")
            np.testing.assert_array_equal(np.load(writer.rank_dir / event["file"]), np.full(4, index))
        self.assertLessEqual(writer._d2h.maximum, 2)
        self.assertEqual(writer._d2h.released, 7)
        self.assertEqual(writer._d2h.active, 0)
        self.assertFalse(writer._worker.is_alive())

    def test_slots_are_held_through_disk_write_and_apply_backpressure(self):
        with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
            writer = RunWriter(self.config)
        self.addCleanup(writer.close)
        entered, unblock, third_finished = threading.Event(), threading.Event(), threading.Event()
        original = writer._write_payload
        def slow_write(*args):
            entered.set()
            if not unblock.wait(3):
                raise RuntimeError("fixture disk timeout")
            original(*args)
        errors = []
        def third():
            try:
                writer.write_tensor(torch.full((4,), 3.))
            except Exception as error:
                errors.append(error)
            finally:
                third_finished.set()
        with patch.object(writer, "_write_payload", side_effect=slow_write):
            writer.write_tensor(torch.ones(4))
            self.assertTrue(entered.wait(2))
            writer.write_tensor(torch.full((4,), 2.))
            worker = threading.Thread(target=third)
            worker.start()
            try:
                self.assertFalse(third_finished.wait(.1))
                self.assertEqual(writer._d2h.active, 2)
            finally:
                unblock.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            writer.flush()
        self.assertFalse(errors)
        self.assertEqual(len(self.records(writer)), 3)

    def test_transfer_failure_is_recorded_and_close_does_not_leave_worker(self):
        with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
            writer = RunWriter(self.config)
        self.addCleanup(writer.close)
        writer._d2h.fail_wait = True
        writer.write_tensor(torch.ones(2))
        writer.close()
        event, = self.records(writer)
        self.assertEqual(event["state"], "error")
        self.assertIn("transfer wait failure", event["reason"])
        self.assertFalse(writer._worker.is_alive())
        self.assertEqual(json.loads((writer.rank_dir / "manifest.json").read_text())["status"], "error")

    def test_nonfinite_observation_finishes_before_boundary_agreement(self):
        config = self.config.to_dict()
        config.update(history_size=2, stop_on_nonfinite=True)
        with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
            probe = PrecisionDebugger(config=config).start()
        self.addCleanup(probe.close)
        for value in (1., 2., 3., float("nan")):
            probe.save(torch.tensor([value]), "watched")
        seen = []
        def coordinate(proposal):
            seen.append(proposal)
            return {**proposal, "accepted": not bool(proposal["error"])}
        result = probe.step_boundary(1, coordinate=coordinate)
        self.assertTrue(seen[0]["stop"])
        self.assertTrue(result["stop"])
        trigger, = probe.writer.manifest["metadata"]["nonfinite_triggers"]
        self.assertEqual([row["event_id"] for row in trigger["history"]], [1, 2])
        self.assertFalse(probe.active)

    def test_statistics_transfer_has_memory_limit_and_cannot_hot_toggle(self):
        self.config.mode = "statistics"
        self.config.max_tensor_bytes = 3
        with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
            probe = PrecisionDebugger(config=self.config).start()
        self.addCleanup(probe.close)
        probe.save(torch.ones(2), "large")
        row, = self.records(probe.writer)
        self.assertEqual(row["state"], "truncated")
        self.assertEqual(probe.writer._d2h.maximum, 0)
        self.assertFalse(probe.request_config_update({"async_d2h": False}))

    def test_queue_handoff_failure_before_and_after_enqueue_has_one_owner(self):
        for enqueue_first in (False, True):
            self.config.run_id = f"handoff_{enqueue_first}"
            with patch("musaprobe.async_transfer.MusaTransferPool", CpuTransportFixture):
                writer = RunWriter(self.config)
            self.addCleanup(writer.close)
            real_put = writer._queue.put
            def interrupted_put(item):
                if enqueue_first:
                    real_put(item)
                raise RuntimeError("controlled interrupted handoff")
            with patch.object(writer._queue, "put", side_effect=interrupted_put):
                with self.assertRaises(AsyncD2HSubmissionError):
                    writer.write_tensor(torch.ones(2))
            writer.close()
            self.assertEqual(writer._d2h.released, 1)
            self.assertEqual(writer._d2h.active, 0)
            self.assertFalse(writer._worker.is_alive())
            records = self.records(writer)
            self.assertEqual(len(records), int(enqueue_first))
            self.assertEqual(len({event["event_id"] for event in records}), len(records))
            self.assertEqual(json.loads((writer.rank_dir / "manifest.json").read_text())["status"], "error")

    def test_acceptance_example_rejects_gradient_capture_errors_despite_correct_training(self):
        from examples.capture_extensions_check import exercise
        original = RunWriter.write_tensor
        def fail_gradients(writer, tensor, **metadata):
            if metadata.get("kind") == "grad_input":
                return writer.write_event({**metadata, "state": "error", "reason": "controlled gradient snapshot failure"})
            return original(writer, tensor, **metadata)
        with patch.object(RunWriter, "write_tensor", fail_gradients):
            with self.assertRaisesRegex(AssertionError, "gradient_snapshots_complete"):
                exercise(Path(self.tmp.name) / "reject_bad_capture", "cpu", torch.float32)
        manifests = list((Path(self.tmp.name) / "reject_bad_capture").rglob("manifest.json"))
        self.assertTrue(manifests)
        self.assertTrue(all(json.loads(path.read_text())["status"] == "error" for path in manifests))


if __name__ == "__main__":
    unittest.main()
