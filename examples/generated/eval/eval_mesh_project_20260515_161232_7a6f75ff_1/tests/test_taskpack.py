import unittest

from taskpack import Planner, Task


class PlannerTests(unittest.TestCase):
    def test_dependency_readiness(self):
        planner = Planner()
        planner.add(Task("design", "Design API"))
        planner.add(Task("build", "Build API", depends_on=("design",)))
        self.assertEqual(planner.ready(), ["design"])
        planner.complete("design")
        self.assertEqual(planner.ready(), ["build"])

    def test_summary(self):
        planner = Planner()
        planner.add(Task("docs", "Write docs"))
        self.assertEqual(planner.summary(), {"total": 1, "done": 0, "open": 1})


if __name__ == "__main__":
    unittest.main()
