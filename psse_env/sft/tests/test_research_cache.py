from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.sft.gates import GateError
from psse_env.sft.research_cache import _verify_weight_manifest


class ResearchCacheTests(unittest.TestCase):
    def test_sharded_manifest_requires_every_nonempty_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            index = snapshot / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"one")
            with self.assertRaisesRegex(GateError, "missing"):
                _verify_weight_manifest(snapshot)
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"two")
            report = _verify_weight_manifest(snapshot)
        self.assertEqual(report["mode"], "sharded_index")
        self.assertEqual(report["weight_files"], 2)
        self.assertEqual(report["weight_bytes"], 6)

    def test_processor_only_snapshot_does_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "processor_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(GateError, "neither"):
                _verify_weight_manifest(snapshot)

    def test_nonempty_single_weight_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "model.safetensors").write_bytes(b"weights")
            report = _verify_weight_manifest(snapshot)
        self.assertEqual(report["mode"], "single_file")
        self.assertEqual(report["weight_files"], 1)


if __name__ == "__main__":
    unittest.main()
