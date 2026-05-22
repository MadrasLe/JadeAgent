"""Reducer hierarchy and rollup tests for Phase E."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import NodeManifest
from jadeagent.mesh import (
    AsyncInMemoryMeshBus,
    AsyncMeshNode,
    MeshRouter,
    MeshTask,
    ReducerNode,
    ShardRuntime,
    ShardSummary,
    ShardSupervisor,
    hillis_steele_reduce,
    hillis_steele_scan,
)


def _worker(node_id: str, *, tenant_id: str, trust_tier: str = "trusted", output: str | None = None) -> AsyncMeshNode:
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
        task_handler=lambda task: output or node_id,
    )


class ReducerTests(unittest.IsolatedAsyncioTestCase):
    def test_hillis_steele_helpers(self):
        values = [1, 2, 3, 4]
        self.assertEqual(hillis_steele_scan(values), [1, 3, 6, 10])
        self.assertEqual(hillis_steele_reduce(values, identity=0), 10)

    async def test_runtime_reduce_rolls_up_shards_and_tenants(self):
        runtime = ShardRuntime()

        sup_acme = ShardSupervisor("sup-acme", tenant_id="acme", capability="summarize")
        sup_beta = ShardSupervisor("sup-beta", tenant_id="beta", capability="summarize")

        sup_acme.register_worker(
            _worker("worker-acme", tenant_id="acme", output="acme-ok"),
            permits=2,
            metadata={"queue_pressure": 1},
        )
        sup_beta.register_worker(
            _worker("worker-beta", tenant_id="beta", output="beta-ok"),
            permits=1,
            metadata={"queue_pressure": 2},
        )

        runtime.register_supervisor(sup_acme)
        runtime.register_supervisor(sup_beta)

        await runtime.submit(MeshTask(capability="summarize", prompt="a", tenant_id="acme"))
        await runtime.submit(MeshTask(capability="summarize", prompt="b", tenant_id="beta"))
        await runtime.run_until_idle()

        summary = runtime.reduce("region-root", region="sa-east")

        self.assertEqual(summary.reducer_id, "region-root")
        self.assertEqual(summary.region, "sa-east")
        self.assertEqual(summary.shard_count, 2)
        self.assertEqual(summary.total_completed, 2)
        self.assertGreaterEqual(summary.total_permits, 3)
        self.assertIn("acme", summary.tenant_budgets)
        self.assertIn("beta", summary.tenant_budgets)
        self.assertEqual(summary.tenant_budgets["acme"].completed, 1)
        self.assertEqual(summary.tenant_budgets["beta"].completed, 1)
        self.assertEqual(len(summary.queue_pressure_prefix), 2)
        self.assertGreater(summary.queue_pressure_total, 0.0)

    def test_reducer_node_merges_child_and_local_rollups(self):
        shard_a = ShardSummary(
            shard_id="acme::summarize",
            supervisor_id="sup-a",
            tenant_id="acme",
            capability="summarize",
            ready_depth=2,
            retry_depth=1,
            inflight=1,
            dead_letter=0,
            completed=3,
            worker_count=2,
            healthy_workers=2,
            busy_workers=1,
            total_permits=4,
            available_permits=3,
            queue_pressure=5.0,
            event_counts={"task_failed": 1, "task_completed": 3},
        )
        shard_b = ShardSummary(
            shard_id="beta::summarize",
            supervisor_id="sup-b",
            tenant_id="beta",
            capability="summarize",
            ready_depth=1,
            retry_depth=0,
            inflight=0,
            dead_letter=1,
            completed=2,
            worker_count=1,
            healthy_workers=1,
            busy_workers=0,
            total_permits=2,
            available_permits=2,
            queue_pressure=2.0,
            event_counts={"task_failed": 1, "policy_denied": 2},
        )

        child = ReducerNode("child")
        child.ingest_shard(shard_a)
        child_summary = child.reduce()

        parent = ReducerNode("parent")
        parent.ingest_child(child_summary)
        parent.ingest_shard(shard_b)
        parent_summary = parent.reduce()

        self.assertEqual(parent_summary.child_count, 1)
        self.assertEqual(parent_summary.shard_count, 1)
        self.assertEqual(parent_summary.total_completed, 5)
        self.assertEqual(parent_summary.total_failed, 2)
        self.assertEqual(parent_summary.event_totals["policy_denied"], 2)
        self.assertEqual(parent_summary.tenant_budgets["acme"].completed, 3)
        self.assertEqual(parent_summary.tenant_budgets["beta"].dead_letter, 1)
        self.assertEqual(parent_summary.queue_pressure_prefix[-1], parent_summary.queue_pressure_total)


if __name__ == "__main__":
    unittest.main()
