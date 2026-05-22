"""Async runtime tests for JadeAgent mesh foundation."""

from __future__ import annotations

import asyncio
import sys
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent.mesh import (
    AsyncInMemoryMeshBus,
    AsyncInMemoryTaskStore,
    AsyncMeshDelegationClient,
    AsyncMeshNode,
    MeshRouter,
    MeshTask,
    TaskResult,
    TaskState,
)


class AsyncRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_task_store_wait_for_terminal(self):
        store = AsyncInMemoryTaskStore()
        task = MeshTask(capability="summarize", prompt="hello", lease_seconds=1.0)

        await store.submit(task)
        claimed = await store.claim_next_available("worker-1", ["summarize"], timeout=0.1)
        self.assertIsNotNone(claimed)

        result = TaskResult(task_id=task.task_id, capability=task.capability, node_id="worker-1")
        result.finalize(TaskState.COMPLETED, output="done")
        await store.complete(task.task_id, "worker-1", result)

        waited = await store.wait_for_terminal(task.task_id, timeout=0.1)
        self.assertIsNotNone(waited)
        self.assertEqual(waited.state, TaskState.COMPLETED)
        self.assertEqual(waited.result.output, "done")

    async def test_async_mesh_node_executes_durable_task_without_polling_loop(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()
        store = AsyncInMemoryTaskStore()

        coordinator = AsyncMeshNode(
            node_id="coordinator",
            capabilities={"delegate"},
            router=router,
            bus=bus,
            task_store=store,
        )

        async def summarize(task: MeshTask) -> str:
            await asyncio.sleep(0.01)
            return f"async-summary:{task.prompt}"

        worker = AsyncMeshNode(
            node_id="worker-a",
            capabilities={"summarize"},
            router=router,
            bus=bus,
            task_store=store,
            task_handler=summarize,
        )

        task_id = await coordinator.submit_task(MeshTask(capability="summarize", prompt="report"))
        worker_step = asyncio.create_task(worker.astep(timeout=1.0))
        result = await coordinator.wait_for_result(task_id, timeout=1.0)
        await worker_step

        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "async-summary:report")

        await coordinator.close()
        await worker.close()

    async def test_async_mesh_transport_returns_result_to_requester(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()

        coordinator = AsyncMeshNode(
            node_id="coordinator",
            capabilities={"delegate"},
            router=router,
            bus=bus,
        )
        worker = AsyncMeshNode(
            node_id="worker-b",
            capabilities={"echo"},
            router=router,
            bus=bus,
            task_handler=lambda task: f"echo:{task.prompt}",
        )

        await worker.start()
        task_id = await coordinator.submit_task(MeshTask(capability="echo", prompt="ping"))
        await asyncio.gather(
            worker.astep(timeout=1.0),
            coordinator.astep(timeout=1.0),
        )
        result = await coordinator.wait_for_result(task_id, timeout=0.1)

        self.assertIsNotNone(result)
        self.assertEqual(result.output, "echo:ping")

        await coordinator.close()
        await worker.close()

    async def test_async_delegation_client_submits_and_waits_for_text(self):
        router = MeshRouter()
        bus = AsyncInMemoryMeshBus()
        store = AsyncInMemoryTaskStore()

        worker = AsyncMeshNode(
            node_id="worker-c",
            capabilities={"classify"},
            router=router,
            bus=bus,
            task_store=store,
            task_handler=lambda task: '{"answer":"positive"}',
        )
        client = AsyncMeshDelegationClient(
            router=router,
            bus=bus,
            task_store=store,
        )

        worker_step = asyncio.create_task(worker.astep(timeout=1.0))
        text = await client.submit_text("classify", "great job", timeout_seconds=1.0)
        await worker_step

        self.assertIn("positive", text)

        await client.close()
        await worker.close()


if __name__ == "__main__":
    unittest.main()
