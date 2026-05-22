"""Regression tests for the distributed JadeAgent runtime roadmap."""

from __future__ import annotations

import sys
import time
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import AccessGrant, MemoryMount, NodeManifest, TaskPolicy, tool
from jadeagent.governance import PolicyBundle
from jadeagent.memory import InMemorySharedMemoryStore, MemoryRouter
from jadeagent.mesh import InMemoryMeshBus, InMemoryTaskStore, MeshNode, MeshRouter, MeshTask


class RecordingAuditSink:
    def __init__(self):
        self.events: list[dict] = []

    def record_event(self, event):
        self.events.append(dict(event))

    def list_events(self, task_id: str | None = None, limit: int = 100):
        events = self.events
        if task_id:
            events = [event for event in events if event.get("task_id") == task_id]
        return events[-limit:]


class FakeSemanticMemory:
    def __init__(self):
        self._entries: list[str] = []

    def remember(self, query: str, k: int = 5) -> list[str]:
        hits = [entry for entry in self._entries if query.lower() in entry.lower()]
        return hits[-k:]

    def memorize(self, content: str, metadata: dict | None = None):
        self._entries.append(content)


@tool(
    description="Fetch remote information",
    effects=["network"],
    resource_refs=["network.outbound"],
)
def fetch_remote() -> str:
    return "ok"


@tool(
    description="Write a quick note",
    effects=["write"],
    write_path_args=["path"],
)
def write_note(path: str, content: str) -> str:
    return f"{path}:{content}"


class RoadmapRuntimeTests(unittest.TestCase):
    def test_access_grants_block_sensitive_tool_effects(self):
        manifest = NodeManifest(
            node_id="net-node",
            constitution=PolicyBundle(enforce_declared_effects=True),
            access=(
                AccessGrant(resource="tool.execute:*", actions=("execute",)),
            ),
        )

        result = fetch_remote.execute({}, node_manifest=manifest)
        self.assertIn("Policy denied", result)
        self.assertIn("not granted", result)

    def test_policy_denial_emits_audit_event(self):
        sink = RecordingAuditSink()
        result = write_note.execute(
            {"path": "summary.txt", "content": "hello"},
            task_policy=TaskPolicy(read_only=True),
            audit_sink=sink,
            execution_context={"task_id": "task-1", "node_id": "writer"},
        )

        self.assertIn("Policy denied", result)
        self.assertTrue(any(event.get("event_type") == "policy_denied" for event in sink.events))

    def test_task_store_submit_claim_complete_and_result_survives(self):
        router = MeshRouter()
        bus = InMemoryMeshBus()
        store = InMemoryTaskStore()

        coordinator = MeshNode(
            node_id="coordinator",
            capabilities={"delegate"},
            router=router,
            bus=bus,
            task_store=store,
        )
        worker = MeshNode(
            node_id="worker-a",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_store=store,
            task_handler=lambda task: f"summary:{task.prompt}",
        )

        task_id = coordinator.submit_task(MeshTask(capability="summarize", prompt="report"))
        bus.run_until_idle()

        result = coordinator.get_result(task_id)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "summary:report")

        events = store.list_events(task_id=task_id)
        event_types = [event.event_type for event in events]
        self.assertIn("task_submitted", event_types)
        self.assertIn("task_claimed", event_types)
        self.assertIn("task_completed", event_types)

        del worker

    def test_task_store_requeues_expired_lease(self):
        store = InMemoryTaskStore()
        task = MeshTask(
            capability="compute",
            prompt="work",
            lease_seconds=0.1,
            max_attempts=2,
        )
        store.submit(task)

        claimed = store.claim_next("worker-1", "compute")
        self.assertIsNotNone(claimed)

        time.sleep(0.15)
        requeued = store.requeue_expired()
        self.assertEqual(requeued, 1)

        reclaimed = store.claim_next("worker-2", "compute")
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.attempts, 2)

    def test_memory_router_enforces_single_writer_and_namespaces(self):
        store = InMemoryTaskStore()
        memory_router = MemoryRouter(
            shared_store=InMemorySharedMemoryStore(),
            task_store=store,
            semantic_factory=lambda tenant_id, scope: FakeSemanticMemory(),
        )

        writer_manifest = NodeManifest(
            node_id="writer",
            tenant_id="tenant-a",
            memory_mounts=(
                MemoryMount(name="private", backend="private_buffer", mode="rw", shared=False),
                MemoryMount(name="scratch", backend="task_scratchpad", mode="rw", shared=True),
                MemoryMount(name="knowledge", backend="semantic_shared", mode="rw", shared=True),
            ),
            access=(
                AccessGrant(resource="memory.read:*", actions=("read",)),
                AccessGrant(resource="memory.write:*", actions=("write",)),
            ),
        )
        reader_manifest = NodeManifest(
            node_id="reader",
            tenant_id="tenant-a",
            memory_mounts=writer_manifest.memory_mounts[1:],
            access=writer_manifest.access,
        )

        task = MeshTask(capability="summarize", prompt="x", tenant_id="tenant-a", memory_scope="scope-a")
        store.submit(task)
        store.claim_next("writer", "summarize")

        memory_router.write_state(
            task.task_id,
            "scratch",
            {"phase": "draft"},
            node_manifest=writer_manifest,
        )
        self.assertEqual(
            memory_router.read_state(task.task_id, "scratch", node_manifest=reader_manifest),
            {"phase": "draft"},
        )

        memory_router.memorize_private(
            "private",
            "only-writer",
            node_manifest=writer_manifest,
        )
        self.assertEqual(
            memory_router.remember_private(
                "private",
                "writer",
                node_manifest=writer_manifest,
            ),
            ["only-writer"],
        )

        memory_router.append_note(task.task_id, "scratch", "writer note", node_manifest=writer_manifest)
        memory_router.append_note(task.task_id, "scratch", "reader note", node_manifest=reader_manifest)
        notes = memory_router.list_notes(task.task_id, "scratch", node_manifest=reader_manifest)
        self.assertEqual(len(notes), 2)

        with self.assertRaises(PermissionError):
            memory_router.write_state(
                task.task_id,
                "scratch",
                {"phase": "tamper"},
                node_manifest=reader_manifest,
            )

        memory_router.memorize(
            "knowledge",
            "alpha memory",
            node_manifest=writer_manifest,
            tenant_id="tenant-a",
            memory_scope="scope-a",
        )
        memory_router.memorize(
            "knowledge",
            "beta memory",
            node_manifest=writer_manifest,
            tenant_id="tenant-b",
            memory_scope="scope-a",
        )
        self.assertEqual(
            memory_router.remember(
                "knowledge",
                "alpha",
                node_manifest=writer_manifest,
                tenant_id="tenant-a",
                memory_scope="scope-a",
            ),
            ["alpha memory"],
        )
        self.assertEqual(
            memory_router.remember(
                "knowledge",
                "beta",
                node_manifest=writer_manifest,
                tenant_id="tenant-a",
                memory_scope="scope-a",
            ),
            [],
        )

    def test_router_filters_by_tenant_trust_and_delegation(self):
        router = MeshRouter()
        router.register_node(
            "global-standard",
            {"summarize"},
            metadata={
                "tenant_id": "",
                "trust_tier": "standard",
                "delegation_allowlist": [],
            },
        )
        router.register_node(
            "tenant-trusted",
            {"summarize"},
            metadata={
                "tenant_id": "acme",
                "trust_tier": "trusted",
                "delegation_allowlist": ["delegator-*"],
            },
        )

        chosen = router.route(
            "summarize",
            tenant_id="acme",
            min_trust_tier="trusted",
            requester="delegator-1",
        )
        self.assertEqual(chosen, "tenant-trusted")

        denied = router.route(
            "summarize",
            tenant_id="acme",
            min_trust_tier="trusted",
            requester="outsider",
        )
        self.assertIsNone(denied)


if __name__ == "__main__":
    unittest.main()
