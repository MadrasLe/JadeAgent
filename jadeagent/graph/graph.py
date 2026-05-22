"""
Graph-based agent orchestration engine.

Build complex agent workflows as directed graphs with:
- Nodes: functions that transform state
- Edges: fixed or conditional transitions
- Cycles: loops with automatic termination
- State: typed dict passed through the graph

Inspired by LangGraph, implemented for JadeAgent.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import START, END, merge_state
from ..state.events import JadeStateEvent
from ..state.manifest import JadeStateManifest, canonical_json_hash
from ..state.snapshot import AgentRuntimeSnapshot, GraphRuntimeSnapshot
from ..state.store import StateStore

logger = logging.getLogger("jadeagent.graph")


@dataclass
class Edge:
    """A transition between two nodes."""
    source: str
    target: str


@dataclass
class ConditionalEdge:
    """A dynamic transition based on state."""
    source: str
    router: Callable[[dict], str]  # fn(state) → next_node_name


class CompiledGraph:
    """
    An executable graph compiled from a StateGraph.

    Use StateGraph.compile() to create this. Then call run() to execute.
    """

    def __init__(
        self,
        nodes: dict[str, Callable],
        edges: list[Edge],
        conditional_edges: list[ConditionalEdge],
        max_iterations: int = 50,
    ):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.max_iterations = max_iterations

        # Build adjacency lookup
        self._fixed_edges: dict[str, str] = {}
        self._cond_edges: dict[str, Callable] = {}

        for e in edges:
            self._fixed_edges[e.source] = e.target
        for ce in conditional_edges:
            self._cond_edges[ce.source] = ce.router

    def _get_next(self, current: str, state: dict) -> str | None:
        """Determine the next node given current node and state."""
        # Conditional edges take priority
        if current in self._cond_edges:
            router = self._cond_edges[current]
            next_node = router(state)
            if next_node not in self.nodes and next_node != END:
                raise ValueError(
                    f"Router from '{current}' returned '{next_node}', "
                    f"which is not a valid node. Available: {list(self.nodes.keys()) + [END]}"
                )
            return next_node

        # Fixed edges
        if current in self._fixed_edges:
            return self._fixed_edges[current]

        return None

    def run(
        self,
        initial_state: dict,
        verbose: bool = False,
        state_store: StateStore | None = None,
        run_id: str | None = None,
        manifest: JadeStateManifest | None = None,
    ) -> dict:
        """
        Execute the graph from START to END.

        Args:
            initial_state: Initial state dict.
            verbose: Print execution trace.

        Returns:
            Final state after reaching END.
        """
        state = dict(initial_state)
        active_run_id = ""
        if state_store is not None:
            manifest = manifest or JadeStateManifest(
                run_id=run_id or uuid.uuid4().hex,
                state_kind="graph",
                capability="graph",
                policy_hash="",
                tool_registry_hash="",
                memory_scope_hash="",
                metadata={
                    "nodes": list(self.nodes.keys()),
                    "edges_hash": canonical_json_hash({
                        "fixed": [(edge.source, edge.target) for edge in self.edges],
                        "conditional": [edge.source for edge in self.conditional_edges],
                    }),
                },
            )
            state_store.create_run(manifest)
            active_run_id = manifest.run_id
            state_store.append_event(active_run_id, JadeStateEvent(
                event_type="graph_started",
                run_id=active_run_id,
                phase="GRAPH_START",
                message="graph execution started",
            ))
        current = self._get_next(START, state)

        if current is None:
            raise ValueError("No edge from START. Add: graph.add_edge(START, 'first_node')")

        visited_count: dict[str, int] = {}

        for iteration in range(self.max_iterations):
            if current == END:
                if verbose:
                    print(f"  [OK] Reached END after {iteration} steps")
                if state_store is not None and active_run_id:
                    snapshot = AgentRuntimeSnapshot(
                        phase="COMPLETED",
                        step=iteration,
                        graph=GraphRuntimeSnapshot(
                            current_node=END,
                            variables=state,
                            iteration=iteration,
                        ),
                    )
                    state_store.save_snapshot(active_run_id, snapshot)
                    state_store.append_event(active_run_id, JadeStateEvent(
                        event_type="graph_completed",
                        run_id=active_run_id,
                        phase="COMPLETED",
                        step=iteration,
                        message="graph execution completed",
                        payload={"snapshot_id": snapshot.snapshot_id},
                    ))
                return state

            if current not in self.nodes:
                raise ValueError(f"Node '{current}' not found in graph")

            # Track visits for cycle detection
            visited_count[current] = visited_count.get(current, 0) + 1

            if verbose:
                print(f"  > [{iteration+1}] Node: {current} (visit #{visited_count[current]})")

            # Execute node
            node_fn = self.nodes[current]
            try:
                result = node_fn(state)
            except Exception as exc:
                if state_store is not None and active_run_id:
                    snapshot = AgentRuntimeSnapshot(
                        phase="FAILED",
                        step=iteration + 1,
                        graph=GraphRuntimeSnapshot(
                            current_node=current,
                            variables=state,
                            iteration=iteration + 1,
                        ),
                        metadata={"error": repr(exc)},
                    )
                    state_store.save_snapshot(active_run_id, snapshot)
                    state_store.append_event(active_run_id, JadeStateEvent(
                        event_type="graph_failed",
                        run_id=active_run_id,
                        phase="FAILED",
                        step=iteration + 1,
                        message=repr(exc),
                        payload={"snapshot_id": snapshot.snapshot_id, "node": current},
                    ))
                raise

            # Merge result into state
            if result is not None and isinstance(result, dict):
                state = merge_state(state, result)

            # Get next node
            next_node = self._get_next(current, state)
            if next_node is None:
                raise ValueError(
                    f"No edge from node '{current}'. "
                    f"Add an edge: graph.add_edge('{current}', 'next_node')"
                )

            if state_store is not None and active_run_id:
                snapshot = AgentRuntimeSnapshot(
                    phase="GRAPH_NODE",
                    step=iteration + 1,
                    graph=GraphRuntimeSnapshot(
                        current_node=current,
                        next_nodes=[next_node],
                        variables=state,
                        iteration=iteration + 1,
                    ),
                )
                state_store.save_snapshot(active_run_id, snapshot)
                state_store.append_event(active_run_id, JadeStateEvent(
                    event_type="checkpoint",
                    run_id=active_run_id,
                    phase="GRAPH_NODE",
                    step=iteration + 1,
                    message="graph checkpoint saved",
                    payload={"node": current, "next_node": next_node, "snapshot_id": snapshot.snapshot_id},
                ))

            current = next_node

        logger.warning(f"Graph hit max iterations ({self.max_iterations})")
        if state_store is not None and active_run_id:
            snapshot = AgentRuntimeSnapshot(
                phase="FAILED",
                step=self.max_iterations,
                graph=GraphRuntimeSnapshot(
                    current_node=current or "",
                    variables=state,
                    iteration=self.max_iterations,
                ),
                metadata={"error": f"Graph hit max iterations ({self.max_iterations})"},
            )
            state_store.save_snapshot(active_run_id, snapshot)
            state_store.append_event(active_run_id, JadeStateEvent(
                event_type="graph_failed",
                run_id=active_run_id,
                phase="FAILED",
                step=self.max_iterations,
                message="graph hit max iterations",
                payload={"snapshot_id": snapshot.snapshot_id},
            ))
        return state


class StateGraph:
    """
    Define a graph of nodes and edges for agent workflows.

    Example:
        from jadeagent.graph import StateGraph, START, END

        class MyState(TypedDict):
            query: str
            result: str

        def search(state):
            return {"result": f"Found info about: {state['query']}"}

        graph = StateGraph()
        graph.add_node("search", search)
        graph.add_edge(START, "search")
        graph.add_edge("search", END)

        result = graph.compile().run({"query": "AI"})
        print(result["result"])

    Cyclic example:
        def should_continue(state):
            return "search" if len(state["results"]) < 3 else END

        graph.add_conditional_edge("analyze", should_continue)
    """

    def __init__(self):
        self.nodes: dict[str, Callable] = {}
        self.edges: list[Edge] = []
        self.conditional_edges: list[ConditionalEdge] = []

    def add_node(self, name: str, fn: Callable[[dict], dict | None]):
        """
        Add a processing node to the graph.

        The function receives the current state dict and optionally
        returns a dict of state updates to merge.

        Args:
            name: Unique node name.
            fn: Function(state) → state_update or None.
        """
        if name in (START, END):
            raise ValueError(f"Cannot use reserved name '{name}'")
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = fn

    def add_edge(self, source: str, target: str):
        """
        Add a fixed transition between nodes.

        Args:
            source: Source node name (or START).
            target: Target node name (or END).
        """
        self._validate_node_ref(source, allow_start=True)
        self._validate_node_ref(target, allow_end=True)
        self.edges.append(Edge(source=source, target=target))

    def add_conditional_edge(self, source: str, router: Callable[[dict], str]):
        """
        Add a dynamic transition based on state.

        The router function receives the current state and returns
        the name of the next node to execute (or END).

        Args:
            source: Source node name.
            router: Function(state) → next_node_name.
        """
        self._validate_node_ref(source, allow_start=True)
        self.conditional_edges.append(
            ConditionalEdge(source=source, router=router)
        )

    def _validate_node_ref(self, name: str, allow_start=False, allow_end=False):
        """Check that a node reference is valid."""
        if name == START and allow_start:
            return
        if name == END and allow_end:
            return
        if name not in self.nodes and name not in (START, END):
            # Lazy validation — node might be added later
            pass

    def compile(self, max_iterations: int = 50) -> CompiledGraph:
        """
        Compile the graph into an executable CompiledGraph.

        Validates the graph structure and returns a runnable graph.

        Args:
            max_iterations: Max iterations before forced termination (cycle safety).

        Returns:
            CompiledGraph ready to run.
        """
        if not self.nodes:
            raise ValueError("Graph has no nodes. Add at least one with add_node().")

        # Check START has an outgoing edge
        has_start = any(
            e.source == START for e in self.edges
        ) or any(
            ce.source == START for ce in self.conditional_edges
        )
        if not has_start:
            raise ValueError("No edge from START. Add: graph.add_edge(START, 'first_node')")

        return CompiledGraph(
            nodes=dict(self.nodes),
            edges=list(self.edges),
            conditional_edges=list(self.conditional_edges),
            max_iterations=max_iterations,
        )
