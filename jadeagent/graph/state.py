"""
Graph state management for JadeAgent's graph engine.

Provides TypedDict-based state containers with merge strategies,
inspired by LangGraph's state management.
"""

from __future__ import annotations

from typing import Any, TypedDict


# Sentinel values for graph control flow
START = "__start__"
END = "__end__"


class GraphState(TypedDict, total=False):
    """
    Base class for graph states. Subclass with your own fields.

    Example:
        class ResearchState(GraphState):
            query: str
            sources: list[str]
            analysis: str
            answer: str
    """
    pass


def merge_state(current: dict, update: dict) -> dict:
    """
    Merge an update into the current state.

    Rules:
    - New keys are added
    - Existing list values are extended (append strategy)
    - Other values are overwritten
    """
    merged = dict(current)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = merged[key] + value  # Append strategy
        else:
            merged[key] = value  # Overwrite strategy
    return merged
