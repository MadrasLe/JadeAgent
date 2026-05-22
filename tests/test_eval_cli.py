"""Eval, timeline, and integrity CLI tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import AgentRuntimeSnapshot, JadeStateEvent, JadeStateManifest, SqliteStateStore
from jadeagent.cli import main as jade_cli_main


class EvalCliTests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = jade_cli_main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_eval_fast_report_timeline_and_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "state.sqlite3"
            output_dir = root / "eval"
            report_path = root / "report.md"

            code, stdout, stderr = self._run_cli([
                "eval",
                "run",
                "--suite",
                "fast",
                "--runs",
                "1",
                "--store",
                str(db_path),
                "--output-dir",
                str(output_dir),
                "--out",
                str(report_path),
                "--json",
            ])
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["aggregate"]["case_runs"], 2)
            self.assertEqual(payload["aggregate"]["failed"], 0)
            self.assertEqual(payload["aggregate"]["duplicate_tool_executions"], 0)
            self.assertEqual(payload["aggregate"]["task_completion_rate"], 1.0)
            self.assertGreater(payload["aggregate"]["total_tokens"], 0)
            self.assertTrue(report_path.exists())

            idempotency = next(
                result for result in payload["results"]
                if result["name"] == "tool_idempotency"
            )
            timeline_path = root / "timeline.html"
            code, stdout, stderr = self._run_cli([
                "state",
                "timeline",
                idempotency["run_id"],
                "--store",
                str(db_path),
                "--html",
                str(timeline_path),
            ])
            self.assertEqual(code, 0, stderr)
            self.assertTrue(timeline_path.exists())
            self.assertIn("JGX Timeline", timeline_path.read_text(encoding="utf-8"))

            code, stdout, stderr = self._run_cli([
                "state",
                "verify",
                idempotency["run_id"],
                "--store",
                str(db_path),
                "--json",
            ])
            self.assertEqual(code, 0, stderr)
            verify_payload = json.loads(stdout)
            self.assertTrue(verify_payload["ok"])
            self.assertEqual(verify_payload["secret_leak_count"], 0)

            report2 = root / "report_from_store.md"
            code, stdout, stderr = self._run_cli([
                "eval",
                "report",
                "--suite",
                "fast",
                "--store",
                str(db_path),
                "--out",
                str(report2),
                "--json",
            ])
            self.assertEqual(code, 0, stderr)
            report_payload = json.loads(stdout)
            self.assertEqual(report_payload["aggregate"]["case_runs"], 2)
            self.assertGreater(report_payload["aggregate"]["restore_latency_ms_p50"], 0)
            self.assertTrue(report2.exists())

    def test_eval_core_includes_raw_call_baseline_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "state.sqlite3"

            code, stdout, stderr = self._run_cli([
                "eval",
                "run",
                "--suite",
                "core",
                "--runs",
                "1",
                "--store",
                str(db_path),
                "--output-dir",
                str(root / "eval"),
                "--out",
                str(root / "report.md"),
                "--json",
            ])
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["aggregate"]["case_runs"], 6)
            self.assertEqual(payload["aggregate"]["raw_baseline_count"], 1)

            raw_case = next(
                result for result in payload["results"]
                if result["name"] == "raw_call_baseline"
            )
            self.assertTrue(raw_case["success"])
            self.assertEqual(raw_case["metrics"]["comparison_target"], "raw_scripted_backend_chat")
            self.assertGreaterEqual(raw_case["metrics"]["state_overhead_ms"], 0)
            self.assertGreater(raw_case["metrics"]["jade_total_tokens"], 0)

    def test_verify_reports_secret_paths_without_printing_secret_values(self):
        secret = "sk-or-v1-testsecretvalue123"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.sqlite3"
            store = SqliteStateStore(db_path)
            store.create_run(JadeStateManifest(
                run_id="secret_run",
                metadata={"token": secret},
            ))
            store.append_event("secret_run", JadeStateEvent(
                event_type="secret_event",
                run_id="secret_run",
                phase="NEW",
                payload={"api_key": secret},
            ))
            store.save_snapshot("secret_run", AgentRuntimeSnapshot(
                phase="COMPLETED",
                metadata={"secret": secret},
            ))
            store.close()

            code, stdout, stderr = self._run_cli([
                "state",
                "verify",
                "secret_run",
                "--store",
                str(db_path),
                "--json",
            ])
            self.assertEqual(code, 1, stderr)
            self.assertNotIn(secret, stdout)
            payload = json.loads(stdout)
            self.assertFalse(payload["ok"])
            self.assertGreaterEqual(payload["secret_leak_count"], 3)
            self.assertTrue(payload["secret_paths"])


if __name__ == "__main__":
    unittest.main()
