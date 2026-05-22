from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_taskpack import Priority, Status, TaskPlanner, export_markdown, load_project, save_project
from mesh_taskpack.cli import build_demo_project


class TaskPackTests(unittest.TestCase):
    def test_add_and_transition_task(self):
        planner = TaskPlanner(name="demo")
        task = planner.add_task("Ship feature", owner="ada", priority=Priority.HIGH)
        self.assertEqual(task.id, "ship-feature")
        self.assertEqual(planner.start_task(task.id).status, Status.DOING)
        self.assertEqual(planner.complete_task(task.id).status, Status.DONE)

    def test_dependency_readiness_and_order(self):
        planner = TaskPlanner(name="deps")
        planner.add_task("Design API", id="design", priority=Priority.HIGH)
        planner.add_task("Implement API", id="impl", depends_on=("design",), priority=Priority.CRITICAL)
        self.assertEqual([task.id for task in planner.ready_tasks()], ["design"])
        planner.complete_task("design")
        self.assertEqual([task.id for task in planner.ready_tasks()], ["impl"])
        self.assertEqual(planner.dependency_order(), ["design", "impl"])

    def test_storage_roundtrip(self):
        planner = build_demo_project()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taskpack.json"
            save_project(planner, path)
            loaded = load_project(path)
        self.assertEqual(loaded.name, "demo")
        self.assertEqual(loaded.dependency_order(), planner.dependency_order())

    def test_markdown_and_risk_report(self):
        planner = TaskPlanner(name="report")
        planner.add_task("Blocked item", id="blocked", owner="bob")
        planner.block_task("blocked")
        text = export_markdown(planner)
        self.assertIn("# report", text)
        self.assertIn("Blocked item", text)
        self.assertEqual(planner.risk_report()["blocked"], ["blocked"])


if __name__ == "__main__":
    unittest.main()
