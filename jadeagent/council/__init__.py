"""Multi-agent orchestration strategies (Compound AI)."""

from .base import Strategy
from .pipeline import Pipeline
from .moa import MixtureOfAgents
from .debate import Debate
from .tot import TreeOfThought

__all__ = [
    "Strategy",
    "Pipeline",
    "MixtureOfAgents",
    "Debate",
    "TreeOfThought",
]
