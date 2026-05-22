"""Shard routing and supervisor tests for Phase B."""

from __future__ import annotations

import asyncio
import sys
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent.mesh import (
    AsyncInMemoryMeshBus,
    AsyncMeshNode,
    MeshRouter,
    MeshTask,
    ShardDirectory,
    ShardRuntime,
    ShardSupervisor,
)


class ShardPhaseBTests(unittest.IsolatedAsyncioTestCase):
    def test_shard_directory_routes_by_tenant_and_capability(self):
        directory = ShardDirectory()
        directory.register_supervisor(
            "sup-acme",
            capabilities={"summarize"},
            tenants={"acme"},
            metadata={"region": "sa-east"},
        )
        directory.register_supervisor(
            "sup-global",
            capabilities={"summarize"},
            metadata={"region": "global"},
        )
        directory.register_supervisor(
            "sup-classify",
            capabilities={"classify"},
        )

        acme_first = directory.route("acme", "summarize")
        acme_second = directory.route("acme", "summarize")
        beta = directory.route("beta", "summarize")
        missing = directory.route("acme", "translate")

        self.assertIsNotNone(acme_first)
        self.assertEqual(acme_first, acme_second)
        self.assertIn(acme_first.supervisor_id, {"sup-acme", "sup-global"})
        self.assertIsNotNone(beta)
        self.assertEqual(beta.supervisor_id, "sup-global")
        self.assertIsNone(missing)

        task = MeshTask(capability="summarize", prompt="hello", tenant_id="acme")
        routed = directory.route_task(task)
        self.assertEqual(routed, acme_first)

    async def test_shard_supervisor_dispatches_to_local_worker(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()
        worker = AsyncMeshNode(
            node_id="worker-sum",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_handler=lambda task: f"summary:{task.prompt}",
        )
        supervisor = ShardSupervisor(
            "sup-acme-summarize",
            tenant_id="acme",
            capability="summarize",
        )
        supervisor.register_worker(worker, permits=2)

        task = MeshTask(capability="summarize", prompt="roadmap", tenant_id="acme")
        await supervisor.submit(task)
        cycles = await supervisor.run_until_idle()

        self.assertGreaterEqual(cycles, 1)
        self.assertIn(task.task_id, supervisor.completed)
        self.assertEqual(supervisor.completed[task.task_id].output, "summary:roadmap")
        self.assertEqual(supervisor.snapshot()["dead_letter"], 0)

    async def test_shard_supervisor_retries_until_success(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()
        attempts = {"count": 0}

        async def flaky(task: MeshTask) -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient")
            await asyncio.sleep(0)
            return f"ok:{task.prompt}"

        worker = AsyncMeshNode(
            node_id="worker-flaky",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_handler=flaky,
        )
        supervisor = ShardSupervisor(
            "sup-acme-summarize",
            tenant_id="acme",
            capability="summarize",
            retry_delay_seconds=0.0,
        )
        supervisor.register_worker(worker)

        task = MeshTask(
            capability="summarize",
            prompt="retry me",
            tenant_id="acme",
            max_attempts=2,
        )
        await supervisor.submit(task)
        await supervisor.run_until_idle()

        self.assertEqual(attempts["count"], 2)
        self.assertIn(task.task_id, supervisor.completed)
        self.assertNotIn(task.task_id, supervisor.dead_letter)
        self.assertEqual(supervisor.completed[task.task_id].output, "ok:retry me")

    async def test_shard_supervisor_dead_letters_after_max_attempts(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()

        async def always_fail(task: MeshTask) -> str:
            await asyncio.sleep(0)
            raise RuntimeError(f"boom:{task.prompt}")

        worker = AsyncMeshNode(
            node_id="worker-fail",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_handler=always_fail,
        )
        supervisor = ShardSupervisor(
            "sup-acme-summarize",
            tenant_id="acme",
            capability="summarize",
            retry_delay_seconds=0.0,
        )
        supervisor.register_worker(worker)

        task = MeshTask(
            capability="summarize",
            prompt="drop me",
            tenant_id="acme",
            max_attempts=1,
        )
        await supervisor.submit(task)
        await supervisor.run_until_idle()

        self.assertNotIn(task.task_id, supervisor.completed)
        self.assertIn(task.task_id, supervisor.dead_letter)
        self.assertIn("boom:drop me", supervisor.dead_letter[task.task_id].error or "")

    async def test_shard_runtime_routes_task_to_registered_supervisor(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()
        runtime = ShardRuntime()

        worker = AsyncMeshNode(
            node_id="worker-runtime",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_handler=lambda task: f"runtime:{task.prompt}",
        )
        supervisor = ShardSupervisor(
            "sup-runtime",
            tenant_id="acme",
            capability="summarize",
        )
        supervisor.register_worker(worker)
        runtime.register_supervisor(supervisor, metadata={"region": "sa-east-1"})

        task = MeshTask(capability="summarize", prompt="through shard", tenant_id="acme")
        assignment = await runtime.submit(task)
        cycles = await runtime.run_until_idle()

        self.assertEqual(assignment.supervisor_id, "sup-runtime")
        self.assertGreaterEqual(cycles["sup-runtime"], 1)
        self.assertIn(task.task_id, supervisor.completed)
        self.assertEqual(supervisor.completed[task.task_id].output, "runtime:through shard")
        snapshot = runtime.snapshot()
        self.assertIn("sup-runtime", snapshot["supervisors"])


if __name__ == "__main__":
    unittest.main()
