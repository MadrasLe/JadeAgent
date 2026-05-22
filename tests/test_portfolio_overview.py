"""Portfolio overview benchmark smoke tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from benchmarks.portfolio_overview import run_portfolio_overview


class PortfolioOverviewTests(unittest.TestCase):
    def test_overview_generates_consolidated_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_portfolio_overview(Path(tmpdir), runtime_runs=1, eval_runs=1)

            self.assertEqual(payload["benchmark"], "portfolio_overview")
            self.assertTrue(Path(payload["json_path"]).exists())
            self.assertTrue(Path(payload["markdown_path"]).exists())
            self.assertIn("runtime", payload["overview"])
            self.assertIn("quality", payload["overview"])
            self.assertIn("challenge", payload["overview"])
            self.assertIn("durable", payload["overview"])
            self.assertIn("jgx_eval", payload["overview"])
            self.assertEqual(payload["overview"]["challenge"]["jade_agent_jgx_pass_rate"], 1.0)
            self.assertEqual(payload["overview"]["durable"]["jade_agent_jgx_recovery_pass_rate"], 1.0)
            self.assertEqual(payload["overview"]["jgx_eval"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
