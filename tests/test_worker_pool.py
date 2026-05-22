"""Local worker index tests for Phase D."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import NodeManifest
from jadeagent.mesh import AsyncInMemoryMeshBus, AsyncMeshNode, LocalWorkerIndex, MeshRouter, MeshTask, ShardSupervisor


def _make_worker(
    node_id: str,
    *,
    trust_tier: str = "standard",
    tenant_id: str = "",
    task_handler=None,
) -> AsyncMeshNode:
    router = MeshRouter()
    bus = AsyncInMemoryMeshBus()
    manifest = NodeManifest(
        node_id=node_id,
        trust_tier=trust_tier,
        tenant_id=tenant_id,
        capabilities=("summarize",),
    )
    return AsyncMeshNode(
        node_id=node_id,
        capabilities={"summarize"},
        router=router,
        bus=bus,
        manifest=manifest,
        task_handler=task_handler or (lambda task: node_id),
    )


class LocalWorkerIndexTests(unittest.IsolatedAsyncioTestCase):
    def test_index_prefers_least_privileged_adequate_worker(self):
        index = LocalWorkerIndex()
        standard = _make_worker("worker-standard", trust_tier="standard")
        trusted = _make_worker("worker-trusted", trust_tier="trusted")
        privileged = _make_worker("worker-privileged", trust_tier="privileged")

        index.register_worker(standard)
        index.register_worker(trusted)
        index.register_worker(privileged)

        task = MeshTask(capability="summarize", prompt="x", min_trust_tier="trusted")
        selected = index.select_worker(task, backlog_depth=3)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.worker.node_id, "worker-trusted")

    def test_index_prefers_more_spare_permits_under_backlog(self):
        index = LocalWorkerIndex()
        small = _make_worker("worker-small", trust_tier="trusted")
        wide = _make_worker("worker-wide", trust_tier="trusted")

        index.register_worker(small, permits=1)
        index.register_worker(wide, permits=4)
        index.reserve("worker-wide")

        task = MeshTask(capability="summarize", prompt="x", min_trust_tier="trusted")
        selected = index.select_worker(task, backlog_depth=12)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.worker.node_id, "worker-wide")

    def test_index_penalizes_queue_pressure_metadata(self):
        index = LocalWorkerIndex()
        calm = _make_worker("worker-calm", trust_tier="trusted")
        hot = _make_worker("worker-hot", trust_tier="trusted")

        index.register_worker(calm, permits=2, metadata={"queue_pressure": 0})
        index.register_worker(hot, permits=2, metadata={"queue_pressure": 6})

        task = MeshTask(capability="summarize", prompt="x", min_trust_tier="trusted")
        first = index.select_worker(task, backlog_depth=2)
        self.assertIsNotNone(first)
        self.assertEqual(first.worker.node_id, "worker-calm")

        index.update_worker("worker-calm", metadata={"queue_pressure": 8})
        index.update_worker("worker-hot", metadata={"queue_pressure": 0})
        second = index.select_worker(task, backlog_depth=2)

        self.assertIsNotNone(second)
        self.assertEqual(second.worker.node_id, "worker-hot")

    async def test_shard_supervisor_uses_local_worker_index(self):
        def make_handler(name: str):
            return lambda task: name

        trusted = _make_worker("worker-trusted", trust_tier="trusted", tenant_id="acme", task_handler=make_handler("trusted"))
        privileged = _make_worker("worker-privileged", trust_tier="privileged", tenant_id="acme", task_handler=make_handler("privileged"))

        supervisor = ShardSupervisor(
            "sup-acme-summarize",
            tenant_id="acme",
            capability="summarize",
        )
        supervisor.register_worker(trusted, permits=1, metadata={"queue_pressure": 0})
        supervisor.register_worker(privileged, permits=1, metadata={"queue_pressure": 0})

        task = MeshTask(
            capability="summarize",
            prompt="route me",
            tenant_id="acme",
            min_trust_tier="trusted",
        )
        await supervisor.submit(task)
        await supervisor.run_until_idle()

        self.assertIn(task.task_id, supervisor.completed)
        self.assertEqual(supervisor.completed[task.task_id].output, "trusted")


if __name__ == "__main__":
    unittest.main()
