"""JGX governed execution state tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import (
    Agent,
    FileStateStore,
    JadeStateEvent,
    JadeStateManifest,
    Session,
    SqliteStateStore,
    tool,
)
from jadeagent.cli import main as jade_cli_main
from jadeagent.backends.base import LLMBackend
from jadeagent.core.types import Message, Response, StreamChunk, ToolCall
from jadeagent.graph import END, START, StateGraph
from jadeagent.mesh import InMemoryMeshBus, MeshNode, MeshRouter, MeshTask
from jadeagent.state.compatibility import validate_restore_compatibility
from jadeagent.state.snapshot import AgentRuntimeSnapshot


class FakeBackend(LLMBackend):
    def __init__(self, responses: list[Response] | None = None):
        self._responses = list(responses or [])

    def chat(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ) -> Response:
        if not self._responses:
            return Response(content="done")
        return self._responses.pop(0)

    def stream(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ):
        if False:
            yield StreamChunk()
        return


@tool(description="Echo text")
def echo_text(text: str) -> str:
    return f"echo:{text}"


class JGXStateTests(unittest.TestCase):
    def test_file_state_store_writes_jgx_capsule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(tmpdir)
            manifest = JadeStateManifest(
                run_id="run_1",
                agent_id="tester",
                tenant_id="tenant_a",
                capability="unit",
            )
            store.create_run(manifest)
            store.append_event("run_1", JadeStateEvent(event_type="run_started"))
            store.save_snapshot("run_1", AgentRuntimeSnapshot(phase="PLANNING", step=0))

            info = store.inspect("run_1")
            self.assertEqual(info["magic"], "JGX1")
            self.assertEqual(info["run_id"], "run_1")
            self.assertEqual(info["latest_phase"], "PLANNING")
            self.assertEqual(info["event_count"], 1)
            self.assertEqual(info["snapshot_count"], 1)

    def test_session_snapshot_round_trips_tool_calls(self):
        backend = FakeBackend()
        session = Session(backend, system_prompt="system")
        session.messages.append(Message.user("hi"))
        session.messages.append(Message.assistant(tool_calls=[
            ToolCall(id="call_1", name="echo_text", arguments={"text": "hi"})
        ]))
        session.add_tool_result("call_1", "echo_text", "echo:hi")

        snapshot = session.snapshot()
        restored = Session.restore(backend, snapshot)

        self.assertEqual(len(restored.messages), 4)
        self.assertEqual(restored.messages[2].tool_calls[0].name, "echo_text")
        self.assertEqual(restored.messages[3].tool_call_id, "call_1")

    def test_agent_run_creates_checkpoints_and_can_restore_latest_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(tmpdir)
            backend = FakeBackend([
                Response(tool_calls=[
                    ToolCall(id="call_1", name="echo_text", arguments={"text": "hello"})
                ]),
                Response(content="final answer"),
            ])
            agent = Agent(
                backend=backend,
                tools=[echo_text],
                verbose=False,
                state_store=store,
            )

            result = agent.run("Use the echo tool")

            self.assertEqual(result.answer, "final answer")
            info = store.inspect(agent.run_id)
            self.assertEqual(info["latest_phase"], "COMPLETED")
            self.assertGreaterEqual(info["snapshot_count"], 5)

            restored = Agent(
                backend=FakeBackend(),
                tools=[echo_text],
                verbose=False,
                state_store=store,
            )
            snapshot = restored.restore_state(agent.run_id)
            self.assertEqual(snapshot.phase, "COMPLETED")
            self.assertTrue(any(message["role"] == "tool" for message in snapshot.messages))
            self.assertTrue(any(message.role == "tool" for message in restored.session.messages))

    def test_restore_compatibility_blocks_tenant_mismatch(self):
        manifest = JadeStateManifest(run_id="r", tenant_id="tenant_a")

        allowed = validate_restore_compatibility(manifest, tenant_id="tenant_a")
        denied = validate_restore_compatibility(manifest, tenant_id="tenant_b")

        self.assertTrue(allowed.allowed)
        self.assertFalse(denied.allowed)
        self.assertIn("tenant_id mismatch", denied.issues[0])

    def test_graph_run_can_checkpoint_to_jgx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(tmpdir)
            graph = StateGraph()
            graph.add_node("double", lambda state: {"value": state["value"] * 2})
            graph.add_edge(START, "double")
            graph.add_edge("double", END)

            result = graph.compile().run(
                {"value": 3},
                state_store=store,
                run_id="graph_run",
            )

            self.assertEqual(result["value"], 6)
            info = store.inspect("graph_run")
            self.assertEqual(info["latest_phase"], "COMPLETED")
            self.assertEqual(info["snapshot_count"], 2)

    def test_mesh_node_can_checkpoint_task_lifecycle_to_jgx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(tmpdir)
            router = MeshRouter()
            bus = InMemoryMeshBus()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                worker = MeshNode(
                    node_id="worker",
                    capabilities={"echo"},
                    router=router,
                    bus=bus,
                    task_handler=lambda task: f"handled:{task.prompt}",
                    state_store=store,
                )
            task = MeshTask(
                capability="echo",
                prompt="hello",
                metadata={"jgx_run_id": "mesh_state"},
            )

            task_id = worker.submit_task(task)
            bus.run_until_idle()
            result = worker.get_result(task_id)

            self.assertIsNotNone(result)
            self.assertEqual(result.output, "handled:hello")
            info = store.inspect("mesh_state")
            self.assertEqual(info["latest_phase"], "COMPLETED")
            self.assertEqual(info["snapshot_count"], 2)

    def test_sqlite_state_store_persists_run_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.sqlite3"
            store = SqliteStateStore(db_path)
            store.create_run(JadeStateManifest(run_id="sqlite_run", agent_id="sqlite_agent"))
            store.append_event("sqlite_run", JadeStateEvent(event_type="run_started", phase="NEW"))
            store.save_snapshot("sqlite_run", AgentRuntimeSnapshot(phase="COMPLETED", step=2))
            store.close()

            reopened = SqliteStateStore(db_path)
            info = reopened.inspect("sqlite_run")
            events = reopened.list_events("sqlite_run")
            latest = reopened.latest_snapshot("sqlite_run")
            reopened.close()

            self.assertEqual(info["latest_phase"], "COMPLETED")
            self.assertEqual(info["snapshot_count"], 1)
            self.assertEqual(info["event_count"], 1)
            self.assertEqual(events[0].event_type, "run_started")
            self.assertEqual(latest.phase, "COMPLETED")

    def test_cli_state_inspect_and_history_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.sqlite3"
            store = SqliteStateStore(db_path)
            store.create_run(JadeStateManifest(run_id="cli_run", agent_id="cli_agent"))
            store.append_event("cli_run", JadeStateEvent(event_type="run_started", phase="NEW"))
            store.save_snapshot("cli_run", AgentRuntimeSnapshot(phase="COMPLETED", step=1))
            store.close()

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = jade_cli_main(["state", "inspect", "cli_run", "--store", db_path])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("run_id: cli_run", stdout.getvalue())
            self.assertIn("latest_phase: COMPLETED", stdout.getvalue())

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = jade_cli_main(["state", "history", "cli_run", "--store", db_path, "--json"])
            self.assertEqual(code, 0, stderr.getvalue())
            payload = __import__("json").loads(stdout.getvalue())
            self.assertEqual(payload[0]["event_type"], "run_started")

    def test_cli_state_list_latest_and_export_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.sqlite3"
            out_path = f"{tmpdir}/exported.jgx"
            store = SqliteStateStore(db_path)
            store.create_run(JadeStateManifest(run_id="cli_more", agent_id="cli_agent"))
            store.append_event("cli_more", JadeStateEvent(event_type="run_started", phase="NEW"))
            store.save_snapshot("cli_more", AgentRuntimeSnapshot(phase="COMPLETED", step=3))
            store.close()

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = jade_cli_main(["state", "list", "--store", db_path])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("cli_more", stdout.getvalue())

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = jade_cli_main(["state", "latest", "cli_more", "--store", db_path])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("phase: COMPLETED", stdout.getvalue())
            self.assertIn("step: 3", stdout.getvalue())

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = jade_cli_main(["state", "export", "cli_more", "--store", db_path, "--out", out_path])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("exported cli_more", stdout.getvalue())

            exported = FileStateStore(tmpdir)
            self.assertEqual(exported.inspect("exported")["latest_phase"], "COMPLETED")

    def test_agent_reuses_recorded_tool_result_for_same_idempotency_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteStateStore(f"{tmpdir}/state.sqlite3")
            calls = {"count": 0}

            @tool(description="Tool with visible side effect")
            def side_effect_tool(value: str) -> str:
                calls["count"] += 1
                return f"side-effect:{value}:{calls['count']}"

            def make_backend():
                return FakeBackend([
                    Response(tool_calls=[
                        ToolCall(id="stable_call", name="side_effect_tool", arguments={"value": "x"})
                    ]),
                    Response(content="done"),
                ])

            first = Agent(
                backend=make_backend(),
                tools=[side_effect_tool],
                verbose=False,
                state_store=store,
                run_id="idem_run",
                max_iterations=1,
            )
            first_result = first.run("call tool")
            self.assertEqual(first_result.answer, "done")
            self.assertEqual(calls["count"], 1)

            second = Agent(
                backend=make_backend(),
                tools=[side_effect_tool],
                verbose=False,
                state_store=store,
                run_id="idem_run",
                max_iterations=1,
            )
            second_result = second.run("call tool")
            events = store.list_events("idem_run", limit=100)
            store.close()

            self.assertEqual(second_result.answer, "done")
            self.assertEqual(calls["count"], 1)
            self.assertTrue(any(event.event_type == "tool_result_recorded" for event in events))
            self.assertTrue(any(event.event_type == "tool_result_reused" for event in events))


if __name__ == "__main__":
    unittest.main()
