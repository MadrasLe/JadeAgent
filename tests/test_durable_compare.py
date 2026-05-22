"""Durable comparison benchmark smoke tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from benchmarks.durable_compare import run_durable_benchmark


class DurableCompareTests(unittest.TestCase):
    def test_jgx_and_langgraph_durable_recover_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_durable_benchmark(Path(tmpdir))

            self.assertEqual(payload["summary"]["jade_agent_jgx"]["recovery_pass_rate"], 1.0)
            self.assertTrue(payload["summary"]["jade_agent_jgx"]["verify_ok"])
            langgraph = payload["summary"]["langgraph_durable_sqlite"]
            if not langgraph.get("skipped"):
                self.assertEqual(langgraph["recovery_pass_rate"], 1.0)
                self.assertGreater(langgraph["checkpoint_count"], 0)


if __name__ == "__main__":
    unittest.main()
