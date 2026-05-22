"""Adversarial challenge benchmark smoke tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from benchmarks.challenge_compare import run_challenge_benchmark


class ChallengeBenchmarkTests(unittest.TestCase):
    def test_jade_jgx_passes_adversarial_capability_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_challenge_benchmark(Path(tmpdir))

            self.assertEqual(payload["summary"]["jade_agent_jgx"]["cases"], 5)
            self.assertEqual(payload["summary"]["jade_agent_jgx"]["pass_rate"], 1.0)
            self.assertTrue(payload["summary"]["jade_agent_jgx"]["jgx_verify_ok"])
            self.assertGreater(payload["summary"]["jade_agent_jgx"]["jgx_event_count"], 0)
            self.assertLess(payload["summary"]["raw_plain"]["pass_rate"], 1.0)
            self.assertTrue(Path(payload["json_path"]).exists())
            self.assertTrue(Path(payload["markdown_path"]).exists())


if __name__ == "__main__":
    unittest.main()
