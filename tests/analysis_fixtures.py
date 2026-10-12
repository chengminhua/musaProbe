import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name, values, *, rank=0, metadata=None, config=None, schema=1, status="complete", world_size=1):
        directory = self.root / name / f"rank_{rank:05d}"
        directory.mkdir(parents=True)
        events = []
        for i, value in enumerate(values):
            array, fields = value if isinstance(value, tuple) else (value, {})
            array = np.asarray(array)
            event = {"schema_version": schema, "event_id": i, "step": 0, "microbatch": 0,
                     "phase": "forward", "module": "model.linear", "call_index": i,
                     "kind": "output", "tensor_path": "$", "op": None,
                     "shape": list(array.shape), "dtype": "torch." + array.dtype.name,
                     "state": "tensor", "file": f"tensor_{i}.npy", "stats": {"numel": array.size}, "stack": []}
            if schema == 2:
                event.update(config_version=0, backward_id=None)
            event.update(fields)
            if event["state"] == "tensor":
                np.save(directory / f"tensor_{i}.npy", array, allow_pickle=False)
            events.append(event)
        (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (directory / "manifest.json").write_text(json.dumps({"schema_version": schema, "rank": rank,
            "world_size": world_size, "status": status, "events_count": len(events),
            "metadata": metadata or {}, "config": config or {}}))
        return directory.parent
