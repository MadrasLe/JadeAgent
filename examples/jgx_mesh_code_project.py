"""Medium mesh workflow that generates, tests, and reviews a Python project.

This example is intentionally practical: a coordinator delegates work to a mesh
of specialized workers, each worker writes or validates part of a generated
project, and every mesh task is captured as a JGX state capsule.

The generated project is timestamped under examples/generated/ so repeated runs
do not overwrite prior artifacts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent import Agent, FileStateStore
from jadeagent.backends import OpenAICompatBackend
from jadeagent.mesh import InMemoryMeshBus, InMemoryTaskStore, MeshNode, MeshRouter, MeshTask


MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
BASE_URL = "https://openrouter.ai/api/v1"
GENERATED_ROOT = ROOT / "examples" / "generated"
REPORT_PATH = ROOT / "docs" / "jgx-mesh-code-project-results.md"


def _read_openrouter_key() -> str:
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    key_path = ROOT / "OPENROUTER.txt"
    if not key_path.exists():
        return ""
    text = key_path.read_text(encoding="utf-8").strip()
    match = re.search(r"sk-or-v1-[A-Za-z0-9_-]+", text)
    return match.group(0) if match else ""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _safe_project_path(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    project_root = project_root.resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {relative}") from exc
    return path


def _make_backend(api_key: str) -> OpenAICompatBackend | None:
    if not api_key:
        return None
    return OpenAICompatBackend(
        model=MODEL,
        base_url=BASE_URL,
        api_key=api_key,
        max_retries=1,
        retry_delay=0.1,
    )


def _json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _run_mesh_task(
    coordinator: MeshNode,
    bus: InMemoryMeshBus,
    capability: str,
    prompt: str,
    *,
    workflow_id: str,
    metadata: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    task = MeshTask(
        capability=capability,
        prompt=prompt,
        requester=coordinator.node_id,
        metadata={
            "workflow_id": workflow_id,
            "jgx_run_id": f"mesh_{capability}_{workflow_id}",
            **dict(metadata or {}),
        },
        lease_seconds=10.0,
        max_attempts=1,
    )
    task_id = coordinator.submit_task(task)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        bus.run_until_idle(max_cycles=100)
        result = coordinator.get_result(task_id)
        if result is not None:
            if not result.success:
                raise RuntimeError(result.error or f"task failed: {capability}")
            try:
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {"raw": result.output}
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {capability}")


def build_workflow() -> dict[str, Any]:
    workflow_id = time.strftime("%Y%m%d_%H%M%S")
    project_root = GENERATED_ROOT / f"mesh_taskpack_{workflow_id}"
    state_root = project_root / ".jgx_runs"
    mesh_state_store = FileStateStore(state_root / "mesh")
    agent_state_store = FileStateStore(state_root / "agents")
    task_store = InMemoryTaskStore()
    router = MeshRouter()
    bus = InMemoryMeshBus()
    api_key = _read_openrouter_key()
    backend = _make_backend(api_key)

    project_root.mkdir(parents=True, exist_ok=True)

    def llm_brief(prompt: str, fallback: str, *, name: str) -> str:
        if backend is None:
            return fallback
        agent = Agent(
            backend=backend,
            name=name,
            system_prompt="You are a concise senior software architect. Avoid markdown tables.",
            state_store=agent_state_store,
            verbose=False,
            max_iterations=1,
            max_tokens=360,
            temperature=0.2,
        )
        try:
            return agent.run(prompt).answer
        except Exception as exc:
            return f"{fallback}\n\nLLM fallback reason: {exc.__class__.__name__}: {exc}"

    def plan_project(task: MeshTask) -> str:
        brief = llm_brief(
            (
                "Create a compact architecture brief for a Python package named mesh_taskpack. "
                "It should manage tasks, dependencies, JSON storage, and a tiny CLI. "
                "Return 5 terse bullets with success criteria."
            ),
            (
                "- Build a Python package for task planning.\n"
                "- Include typed dataclasses, status transitions, dependency checks, JSON storage, and CLI.\n"
                "- Include unit tests for lifecycle, dependency ordering, and persistence.\n"
                "- Keep dependencies to the standard library.\n"
                "- Make the result inspectable and reusable."
            ),
            name="mesh_project_architect",
        )
        _write(project_root / "docs" / "architecture_brief.md", brief)
        return _json_result(worker="planner", files=["docs/architecture_brief.md"], brief_chars=len(brief))

    def write_domain(task: MeshTask) -> str:
        _write(
            _safe_project_path(project_root, "mesh_taskpack/models.py"),
            r'''
            """Domain models for mesh_taskpack."""

            from __future__ import annotations

            import time
            from dataclasses import dataclass, field
            from enum import Enum
            from typing import Any


            class TaskValidationError(ValueError):
                """Raised when task data is invalid."""


            class Status(str, Enum):
                TODO = "todo"
                DOING = "doing"
                BLOCKED = "blocked"
                DONE = "done"


            class Priority(int, Enum):
                LOW = 1
                MEDIUM = 2
                HIGH = 3
                CRITICAL = 4


            @dataclass(frozen=True)
            class Task:
                id: str
                title: str
                description: str = ""
                owner: str = "unassigned"
                priority: Priority = Priority.MEDIUM
                status: Status = Status.TODO
                tags: tuple[str, ...] = ()
                depends_on: tuple[str, ...] = ()
                created_at: float = field(default_factory=time.time)
                updated_at: float = field(default_factory=time.time)

                def __post_init__(self) -> None:
                    if not self.id or not self.id.strip():
                        raise TaskValidationError("task id is required")
                    if not self.title or not self.title.strip():
                        raise TaskValidationError("task title is required")
                    object.__setattr__(self, "id", self.id.strip())
                    object.__setattr__(self, "title", self.title.strip())
                    object.__setattr__(self, "owner", (self.owner or "unassigned").strip())
                    object.__setattr__(self, "tags", tuple(str(tag).strip() for tag in self.tags if str(tag).strip()))
                    object.__setattr__(
                        self,
                        "depends_on",
                        tuple(str(dep).strip() for dep in self.depends_on if str(dep).strip()),
                    )
                    if isinstance(self.priority, int) and not isinstance(self.priority, Priority):
                        object.__setattr__(self, "priority", Priority(self.priority))
                    if isinstance(self.status, str):
                        object.__setattr__(self, "status", Status(self.status))

                def with_status(self, status: Status | str) -> "Task":
                    return Task(
                        id=self.id,
                        title=self.title,
                        description=self.description,
                        owner=self.owner,
                        priority=self.priority,
                        status=Status(status),
                        tags=self.tags,
                        depends_on=self.depends_on,
                        created_at=self.created_at,
                        updated_at=time.time(),
                    )

                def is_ready(self, completed_ids: set[str]) -> bool:
                    return self.status == Status.TODO and all(dep in completed_ids for dep in self.depends_on)

                def to_dict(self) -> dict[str, Any]:
                    return {
                        "id": self.id,
                        "title": self.title,
                        "description": self.description,
                        "owner": self.owner,
                        "priority": int(self.priority),
                        "status": self.status.value,
                        "tags": list(self.tags),
                        "depends_on": list(self.depends_on),
                        "created_at": self.created_at,
                        "updated_at": self.updated_at,
                    }

                @classmethod
                def from_dict(cls, data: dict[str, Any]) -> "Task":
                    return cls(
                        id=str(data["id"]),
                        title=str(data["title"]),
                        description=str(data.get("description", "")),
                        owner=str(data.get("owner", "unassigned")),
                        priority=Priority(int(data.get("priority", Priority.MEDIUM))),
                        status=Status(str(data.get("status", Status.TODO.value))),
                        tags=tuple(data.get("tags", ())),
                        depends_on=tuple(data.get("depends_on", ())),
                        created_at=float(data.get("created_at", time.time())),
                        updated_at=float(data.get("updated_at", time.time())),
                    )
            ''',
        )
        _write(
            _safe_project_path(project_root, "mesh_taskpack/__init__.py"),
            r'''
            """Task planning primitives generated by a JadeAgent mesh workflow."""

            from .models import Priority, Status, Task, TaskValidationError
            from .planner import TaskPlanner
            from .storage import export_markdown, load_project, save_project

            __all__ = [
                "Priority",
                "Status",
                "Task",
                "TaskPlanner",
                "TaskValidationError",
                "export_markdown",
                "load_project",
                "save_project",
            ]
            ''',
        )
        return _json_result(worker="domain", files=["mesh_taskpack/models.py", "mesh_taskpack/__init__.py"])

    def write_engine(task: MeshTask) -> str:
        _write(
            _safe_project_path(project_root, "mesh_taskpack/planner.py"),
            r'''
            """Planning engine for mesh_taskpack."""

            from __future__ import annotations

            import re
            from collections import defaultdict
            from dataclasses import dataclass, field
            from typing import Iterable

            from .models import Priority, Status, Task, TaskValidationError


            def _slugify(value: str) -> str:
                slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
                return slug or "task"


            @dataclass
            class TaskPlanner:
                name: str = "taskpack"
                tasks: dict[str, Task] = field(default_factory=dict)

                def add_task(
                    self,
                    title: str,
                    *,
                    id: str | None = None,
                    description: str = "",
                    owner: str = "unassigned",
                    priority: Priority | int = Priority.MEDIUM,
                    tags: Iterable[str] = (),
                    depends_on: Iterable[str] = (),
                ) -> Task:
                    task_id = id or self._next_id(title)
                    missing = [dep for dep in depends_on if dep not in self.tasks]
                    if missing:
                        raise TaskValidationError(f"unknown dependencies: {', '.join(missing)}")
                    task = Task(
                        id=task_id,
                        title=title,
                        description=description,
                        owner=owner,
                        priority=Priority(priority),
                        tags=tuple(tags),
                        depends_on=tuple(depends_on),
                    )
                    if task.id in self.tasks:
                        raise TaskValidationError(f"duplicate task id: {task.id}")
                    self.tasks[task.id] = task
                    return task

                def _next_id(self, title: str) -> str:
                    base = _slugify(title)
                    candidate = base
                    index = 2
                    while candidate in self.tasks:
                        candidate = f"{base}-{index}"
                        index += 1
                    return candidate

                def get(self, task_id: str) -> Task:
                    try:
                        return self.tasks[task_id]
                    except KeyError as exc:
                        raise TaskValidationError(f"unknown task id: {task_id}") from exc

                def set_status(self, task_id: str, status: Status | str) -> Task:
                    task = self.get(task_id)
                    if Status(status) == Status.DONE:
                        missing = [dep for dep in task.depends_on if self.tasks[dep].status != Status.DONE]
                        if missing:
                            raise TaskValidationError(f"cannot complete with unfinished dependencies: {missing}")
                    updated = task.with_status(status)
                    self.tasks[task_id] = updated
                    return updated

                def start_task(self, task_id: str) -> Task:
                    return self.set_status(task_id, Status.DOING)

                def block_task(self, task_id: str) -> Task:
                    return self.set_status(task_id, Status.BLOCKED)

                def complete_task(self, task_id: str) -> Task:
                    return self.set_status(task_id, Status.DONE)

                def completed_ids(self) -> set[str]:
                    return {task.id for task in self.tasks.values() if task.status == Status.DONE}

                def ready_tasks(self) -> list[Task]:
                    completed = self.completed_ids()
                    return sorted(
                        [task for task in self.tasks.values() if task.is_ready(completed)],
                        key=lambda task: (-int(task.priority), task.created_at),
                    )

                def list_by_status(self, status: Status | str) -> list[Task]:
                    wanted = Status(status)
                    return [task for task in self.tasks.values() if task.status == wanted]

                def workload_by_owner(self) -> dict[str, int]:
                    counts: dict[str, int] = defaultdict(int)
                    for task in self.tasks.values():
                        if task.status != Status.DONE:
                            counts[task.owner] += 1
                    return dict(sorted(counts.items()))

                def dependency_order(self) -> list[str]:
                    visiting: set[str] = set()
                    visited: set[str] = set()
                    ordered: list[str] = []

                    def visit(task_id: str) -> None:
                        if task_id in visited:
                            return
                        if task_id in visiting:
                            raise TaskValidationError(f"cycle detected at {task_id}")
                        visiting.add(task_id)
                        for dep in self.get(task_id).depends_on:
                            visit(dep)
                        visiting.remove(task_id)
                        visited.add(task_id)
                        ordered.append(task_id)

                    for task_id in self.tasks:
                        visit(task_id)
                    return ordered

                def risk_report(self) -> dict[str, list[str]]:
                    blocked = [task.id for task in self.tasks.values() if task.status == Status.BLOCKED]
                    waiting = [
                        task.id
                        for task in self.tasks.values()
                        if task.status == Status.TODO and task.depends_on and not task.is_ready(self.completed_ids())
                    ]
                    overloaded = [owner for owner, count in self.workload_by_owner().items() if count >= 3]
                    return {
                        "blocked": blocked,
                        "waiting_on_dependencies": waiting,
                        "overloaded_owners": overloaded,
                    }

                def to_dict(self) -> dict:
                    return {
                        "name": self.name,
                        "tasks": [self.tasks[task_id].to_dict() for task_id in self.dependency_order()],
                    }

                @classmethod
                def from_dict(cls, data: dict) -> "TaskPlanner":
                    planner = cls(name=str(data.get("name", "taskpack")))
                    for raw_task in data.get("tasks", []):
                        task = Task.from_dict(raw_task)
                        planner.tasks[task.id] = task
                    planner.dependency_order()
                    return planner
            ''',
        )
        return _json_result(worker="engine", files=["mesh_taskpack/planner.py"])

    def write_storage_cli(task: MeshTask) -> str:
        _write(
            _safe_project_path(project_root, "mesh_taskpack/storage.py"),
            r'''
            """Persistence helpers for mesh_taskpack."""

            from __future__ import annotations

            import json
            from pathlib import Path

            from .models import Status
            from .planner import TaskPlanner


            def save_project(planner: TaskPlanner, path: str | Path) -> Path:
                output = Path(path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(planner.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
                return output


            def load_project(path: str | Path) -> TaskPlanner:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                return TaskPlanner.from_dict(data)


            def export_markdown(planner: TaskPlanner) -> str:
                lines = [f"# {planner.name}", ""]
                for status in Status:
                    tasks = planner.list_by_status(status)
                    lines.append(f"## {status.value}")
                    if not tasks:
                        lines.append("- none")
                    for task in tasks:
                        deps = f" deps={','.join(task.depends_on)}" if task.depends_on else ""
                        lines.append(f"- [{task.id}] {task.title} owner={task.owner} priority={int(task.priority)}{deps}")
                    lines.append("")
                return "\n".join(lines).rstrip() + "\n"
            ''',
        )
        _write(
            _safe_project_path(project_root, "mesh_taskpack/cli.py"),
            r'''
            """Small command line interface for mesh_taskpack."""

            from __future__ import annotations

            import argparse

            from .models import Priority, Status
            from .planner import TaskPlanner
            from .storage import export_markdown, load_project, save_project


            def build_demo_project() -> TaskPlanner:
                planner = TaskPlanner(name="demo")
                planner.add_task("Write domain model", owner="domain", priority=Priority.HIGH)
                planner.add_task("Build planner", owner="engine", priority=Priority.HIGH, depends_on=("write-domain-model",))
                planner.add_task("Add tests", owner="qa", priority=Priority.CRITICAL, depends_on=("build-planner",))
                return planner


            def main(argv: list[str] | None = None) -> int:
                parser = argparse.ArgumentParser(prog="mesh-taskpack")
                parser.add_argument("--file", default="taskpack.json")
                sub = parser.add_subparsers(dest="command", required=True)
                sub.add_parser("demo")
                sub.add_parser("summary")
                add = sub.add_parser("add")
                add.add_argument("title")
                add.add_argument("--owner", default="unassigned")
                add.add_argument("--priority", type=int, default=int(Priority.MEDIUM))
                done = sub.add_parser("done")
                done.add_argument("task_id")
                args = parser.parse_args(argv)

                if args.command == "demo":
                    planner = build_demo_project()
                    save_project(planner, args.file)
                    print(export_markdown(planner))
                    return 0

                try:
                    planner = load_project(args.file)
                except FileNotFoundError:
                    planner = TaskPlanner()

                if args.command == "add":
                    planner.add_task(args.title, owner=args.owner, priority=Priority(args.priority))
                    save_project(planner, args.file)
                    print(export_markdown(planner))
                    return 0

                if args.command == "done":
                    planner.complete_task(args.task_id)
                    save_project(planner, args.file)
                    print(export_markdown(planner))
                    return 0

                if args.command == "summary":
                    print(export_markdown(planner))
                    risks = planner.risk_report()
                    if any(risks.values()):
                        print("Risks:", risks)
                    return 0

                raise AssertionError(f"unhandled command: {args.command}")


            if __name__ == "__main__":
                raise SystemExit(main())
            ''',
        )
        return _json_result(worker="storage_cli", files=["mesh_taskpack/storage.py", "mesh_taskpack/cli.py"])

    def write_tests_docs(task: MeshTask) -> str:
        _write(
            _safe_project_path(project_root, "tests/test_taskpack.py"),
            r'''
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
            ''',
        )
        _write(
            _safe_project_path(project_root, "README.md"),
            r'''
            # mesh_taskpack

            `mesh_taskpack` is a small standard-library Python package generated
            by a JadeAgent mesh workflow. It models tasks, dependencies, owners,
            status transitions, JSON persistence, markdown export, and a tiny CLI.

            ## Quick Start

            ```bash
            python -m mesh_taskpack.cli --file demo.json demo
            python -m mesh_taskpack.cli --file demo.json summary
            ```

            ## Development

            ```bash
            python -m unittest discover -s tests -p "test_*.py"
            ```
            ''',
        )
        _write(
            _safe_project_path(project_root, "pyproject.toml"),
            r'''
            [build-system]
            requires = ["setuptools>=68.0", "wheel"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "mesh-taskpack"
            version = "0.1.0"
            description = "Task planning package generated by a JadeAgent mesh workflow"
            requires-python = ">=3.10"
            authors = [{name = "JadeAgent Mesh"}]

            [tool.setuptools.packages.find]
            include = ["mesh_taskpack*"]
            ''',
        )
        return _json_result(worker="tests_docs", files=["tests/test_taskpack.py", "README.md", "pyproject.toml"])

    def run_project_tests(task: MeshTask) -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        payload = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": "python -m unittest discover -s tests -p test_*.py",
        }
        _write(project_root / "test_result.json", json.dumps(payload, indent=2, ensure_ascii=True))
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        return _json_result(worker="test_runner", files=["test_result.json"], returncode=completed.returncode)

    def review_project(task: MeshTask) -> str:
        file_list = sorted(
            str(path.relative_to(project_root)).replace("\\", "/")
            for path in project_root.rglob("*")
            if path.is_file() and ".jgx_runs" not in path.parts
        )
        test_result_path = project_root / "test_result.json"
        test_result = json.loads(test_result_path.read_text(encoding="utf-8"))
        fallback = (
            "Review: project structure is coherent, tests pass, and the package uses only the standard library. "
            "Next improvement: add richer CLI commands and package metadata."
        )
        review = llm_brief(
            (
                "Review this generated Python project in 4 terse bullets. "
                f"Files: {file_list}. Test return code: {test_result['returncode']}. "
                "Mention one limitation."
            ),
            fallback,
            name="mesh_project_reviewer",
        )
        _write(project_root / "docs" / "review.md", review)
        return _json_result(worker="reviewer", files=["docs/review.md"], review_chars=len(review))

    coordinator = MeshNode(
        node_id="mesh_project_coordinator",
        capabilities={"coordinate"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=mesh_state_store,
        task_handler=lambda task: "coordinator",
    )

    worker_specs = [
        ("mesh_planner", "plan_project", plan_project),
        ("mesh_domain_writer", "write_domain", write_domain),
        ("mesh_engine_writer", "write_engine", write_engine),
        ("mesh_storage_cli_writer", "write_storage_cli", write_storage_cli),
        ("mesh_tests_docs_writer", "write_tests_docs", write_tests_docs),
        ("mesh_test_runner", "run_project_tests", run_project_tests),
        ("mesh_reviewer", "review_project", review_project),
    ]
    workers = [
        MeshNode(
            node_id=node_id,
            capabilities={capability},
            router=router,
            bus=bus,
            task_handler=handler,
            task_store=task_store,
            state_store=mesh_state_store,
        )
        for node_id, capability, handler in worker_specs
    ]

    steps = [
        ("plan_project", "Create architecture brief"),
        ("write_domain", "Write domain models"),
        ("write_engine", "Write planning engine"),
        ("write_storage_cli", "Write storage and CLI"),
        ("write_tests_docs", "Write tests and docs"),
        ("run_project_tests", "Run generated project tests"),
        ("review_project", "Review generated project"),
    ]

    step_results: list[dict[str, Any]] = []
    for capability, prompt in steps:
        result = _run_mesh_task(
            coordinator,
            bus,
            capability,
            prompt,
            workflow_id=workflow_id,
            metadata={"project_root": str(project_root)},
            timeout_seconds=90.0 if capability in {"plan_project", "review_project"} else 30.0,
        )
        step_results.append({"capability": capability, "result": result})

    state_inspect = {
        path.name[:-4]: mesh_state_store.inspect(path.name[:-4])
        for path in sorted((state_root / "mesh").glob("*.jgx"))
        if path.is_dir()
    }
    agent_inspect = {
        path.name[:-4]: agent_state_store.inspect(path.name[:-4])
        for path in sorted((state_root / "agents").glob("*.jgx"))
        if path.is_dir()
    }

    test_payload = json.loads((project_root / "test_result.json").read_text(encoding="utf-8"))
    report = {
        "workflow_id": workflow_id,
        "project_root": str(project_root),
        "model": MODEL if backend is not None else "fallback-no-openrouter-key",
        "steps": step_results,
        "mesh_capsules": state_inspect,
        "agent_capsules": agent_inspect,
        "test_returncode": test_payload["returncode"],
        "test_stdout": test_payload["stdout"],
        "test_stderr": test_payload["stderr"],
    }
    _write(project_root / "workflow_result.json", json.dumps(report, indent=2, ensure_ascii=True))
    _write_report(report)

    return report


def _write_report(report: dict[str, Any]) -> None:
    mesh_rows = []
    for run_id, info in sorted(report["mesh_capsules"].items()):
        mesh_rows.append(
            f"| `{run_id}` | `{info['agent_id']}` | `{info['capability']}` | "
            f"`{info['latest_phase']}` | {info['snapshot_count']} | {info['event_count']} |"
        )
    agent_rows = []
    for run_id, info in sorted(report["agent_capsules"].items()):
        agent_rows.append(
            f"| `{run_id}` | `{info['agent_id']}` | `{info['latest_phase']}` | "
            f"{info['snapshot_count']} | {info['event_count']} |"
        )

    mesh_steps = "\n".join(
        f"- `{step['capability']}` -> `{step['result'].get('worker', 'unknown')}`"
        for step in report["steps"]
    )
    content = "\n".join([
        "# JGX Mesh Code Project Results",
        "",
        "Date: 2026-05-07",
        "",
        "This report records a medium mesh workflow that generated, tested, reviewed,",
        "and checkpointed a real Python project.",
        "",
        "## Run",
        "",
        f"- Workflow id: `{report['workflow_id']}`",
        f"- Generated project: `{report['project_root']}`",
        f"- Model: `{report['model']}`",
        f"- Test return code: `{report['test_returncode']}`",
        "",
        "## Mesh Steps",
        "",
        mesh_steps,
        "",
        "## Mesh JGX Capsules",
        "",
        "| Run id | Worker | Capability | Phase | Snapshots | Events |",
        "|---|---|---|---:|---:|---:|",
        "\n".join(mesh_rows),
        "",
        "## LLM Agent JGX Capsules",
        "",
        "| Run id | Agent | Phase | Snapshots | Events |",
        "|---|---|---:|---:|---:|",
        "\n".join(agent_rows) if agent_rows else "| none | none | none | 0 | 0 |",
        "",
        "## Generated Project Tests",
        "",
        "```text",
        f"{report['test_stdout']}{report['test_stderr']}".rstrip(),
        "```",
        "",
        "## Result",
        "",
        "The workflow produced a medium-sized Python package with domain models,",
        "planning logic, JSON persistence, CLI, README, tests, a test result artifact,",
        "a model-assisted architecture brief, and a model-assisted review.",
        "",
    ])
    _write(REPORT_PATH, content)


if __name__ == "__main__":
    result = build_workflow()
    print("MESH_CODE_WORKFLOW=" + json.dumps({
        "workflow_id": result["workflow_id"],
        "project_root": result["project_root"],
        "model": result["model"],
        "mesh_capsules": len(result["mesh_capsules"]),
        "agent_capsules": len(result["agent_capsules"]),
        "test_returncode": result["test_returncode"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=True, sort_keys=True))
