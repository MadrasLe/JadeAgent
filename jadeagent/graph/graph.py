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
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import START, END, merge_state

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

    def run(self, initial_state: dict, verbose: bool = False) -> dict:
        """
        Execute the graph from START to END.

        Args:
            initial_state: Initial state dict.
            verbose: Print execution trace.

        Returns:
            Final state after reaching END.
        """
        state = dict(initial_state)
        current = self._get_next(START, state)

        if current is None:
            raise ValueError("No edge from START. Add: graph.add_edge(START, 'first_node')")

        visited_count: dict[str, int] = {}

        for iteration in range(self.max_iterations):
            if current == END:
                if verbose:
                    print(f"  [OK] Reached END after {iteration} steps")
                return state

            if current not in self.nodes:
                raise ValueError(f"Node '{current}' not found in graph")

            # Track visits for cycle detection
            visited_count[current] = visited_count.get(current, 0) + 1

            if verbose:
                print(f"  > [{iteration+1}] Node: {current} (visit #{visited_count[current]})")

            # Execute node
            node_fn = self.nodes[current]
            result = node_fn(state)

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

            current = next_node

        logger.warning(f"Graph hit max iterations ({self.max_iterations})")
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
