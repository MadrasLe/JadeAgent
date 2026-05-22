"""
Shard runtime orchestration for Phase B control-plane routing.
"""

from __future__ import annotations

from typing import Any

from .protocol import MeshTask
from .reducer import ReducerNode, ReductionSummary, ShardSummary
from .sharding import ShardAssignment, ShardDirectory
from .supervisor import ShardSupervisor


class ShardRuntime:
    """
    Thin orchestration layer that ties shard ownership to local supervisors.

    This keeps Phase B explicit: `task -> shard assignment -> supervisor queue`.
    It does not replace the legacy mesh path; it offers the new control-plane
    path as an opt-in runtime surface.
    """

    def __init__(self, directory: ShardDirectory | None = None):
        self.directory = directory or ShardDirectory()
        self._supervisors: dict[str, ShardSupervisor] = {}

    def register_supervisor(
        self,
        supervisor: ShardSupervisor,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ):
        self._supervisors[supervisor.supervisor_id] = supervisor
        merged_metadata = {
            "tenant_id": supervisor.tenant_id,
            "capability": supervisor.capability,
            **dict(metadata or {}),
        }
        tenants = {supervisor.tenant_id} if supervisor.tenant_id else None
        self.directory.register_supervisor(
            supervisor.supervisor_id,
            capabilities={supervisor.capability},
            tenants=tenants,
            weight=weight,
            metadata=merged_metadata,
        )

    def unregister_supervisor(self, supervisor_id: str):
        self._supervisors.pop(supervisor_id, None)
        self.directory.unregister_supervisor(supervisor_id)

    def get_supervisor(self, supervisor_id: str) -> ShardSupervisor | None:
        return self._supervisors.get(supervisor_id)

    def route(self, task: MeshTask) -> ShardAssignment | None:
        return self.directory.route_task(task)

    def resolve(self, task: MeshTask) -> tuple[ShardAssignment, ShardSupervisor] | None:
        assignment = self.route(task)
        if assignment is None:
            return None
        supervisor = self._supervisors.get(assignment.supervisor_id)
        if supervisor is None:
            return None
        return assignment, supervisor

    async def submit(self, task: MeshTask) -> ShardAssignment:
        resolved = self.resolve(task)
        if resolved is None:
            raise LookupError(
                f"No shard supervisor available for tenant '{task.tenant_id}' "
                f"and capability '{task.capability}'."
            )
        assignment, supervisor = resolved
        await supervisor.submit(task)
        return assignment

    async def run_until_idle(self, max_cycles_per_supervisor: int = 1000) -> dict[str, int]:
        cycles: dict[str, int] = {}
        for supervisor in sorted(self._supervisors.values(), key=lambda item: item.supervisor_id):
            cycles[supervisor.supervisor_id] = await supervisor.run_until_idle(max_cycles=max_cycles_per_supervisor)
        return cycles

    def collect_shard_summaries(self) -> list[ShardSummary]:
        return [
            ShardSummary.from_supervisor_snapshot(supervisor.snapshot())
            for supervisor in sorted(self._supervisors.values(), key=lambda item: item.supervisor_id)
        ]

    def reduce(self, reducer_id: str = "root", *, region: str = "") -> ReductionSummary:
        reducer = ReducerNode(reducer_id, region=region)
        for summary in self.collect_shard_summaries():
            reducer.ingest_shard(summary)
        return reducer.reduce()

    def snapshot(self) -> dict[str, Any]:
        return {
            "directory": self.directory.snapshot(),
            "supervisors": {
                supervisor_id: supervisor.snapshot()
                for supervisor_id, supervisor in sorted(self._supervisors.items())
            },
        }
