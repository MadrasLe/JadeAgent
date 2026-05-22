"""Lease wheel and deadline-aware expiry tests for Phase C."""

from __future__ import annotations

import asyncio
import sys
import time
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent.mesh import AsyncInMemoryTaskStore, InMemoryTaskStore, LeaseDeadlineIndex, MeshTask


class LeaseWheelTests(unittest.IsolatedAsyncioTestCase):
    def test_lease_deadline_index_handles_renew_and_clear(self):
        index = LeaseDeadlineIndex()
        now = time.time()

        first = index.upsert("task-1", "worker-a", now + 0.2)
        renewed = index.upsert("task-1", "worker-a", now + 0.5)
        second = index.upsert("task-2", "worker-b", now + 0.1)

        self.assertEqual(first.version, 1)
        self.assertEqual(renewed.version, 2)
        self.assertEqual(index.next_deadline(), second.deadline)

        expired = index.pop_expired(now + 0.15)
        self.assertEqual([lease.task_id for lease in expired], ["task-2"])

        index.discard("task-1")
        self.assertEqual(index.pop_expired(now + 1.0), [])

    async def test_async_task_store_waits_until_lease_deadline_then_requeues(self):
        store = AsyncInMemoryTaskStore()
        task = MeshTask(
            capability="summarize",
            prompt="late lease",
            lease_seconds=0.12,
            max_attempts=2,
        )

        await store.submit(task)
        claimed = await store.claim_next("worker-a", "summarize")
        self.assertIsNotNone(claimed)

        started_at = time.time()
        reclaimed = await store.claim_next_available("worker-b", ["summarize"], timeout=0.5)
        elapsed = time.time() - started_at

        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.task_id, task.task_id)
        self.assertEqual(reclaimed.lease_owner, "worker-b")
        self.assertGreaterEqual(elapsed, 0.08)
        self.assertLess(elapsed, 0.4)

        events = await store.list_events(task.task_id)
        self.assertTrue(any(event.event_type == "lease_expired" for event in events))

    def test_sync_task_store_requeues_from_deadline_index(self):
        store = InMemoryTaskStore()
        task = MeshTask(
            capability="compute",
            prompt="sync",
            lease_seconds=0.1,
            max_attempts=2,
        )

        store.submit(task)
        store.claim_next("worker-a", "compute")
        time.sleep(0.12)

        requeued = store.requeue_expired()
        reclaimed = store.claim_next("worker-b", "compute")

        self.assertEqual(requeued, 1)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.lease_owner, "worker-b")


if __name__ == "__main__":
    unittest.main()
