"""
Base class for multi-agent council strategies.

A strategy orchestrates multiple Agent instances to solve a task
collaboratively, using different patterns (MoA, Debate, ToT, Pipeline).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("jadeagent.council")


class Strategy(ABC):
    """
    Abstract base for multi-agent orchestration strategies.

    All strategies receive a task string and return a final answer.
    """

    @abstractmethod
    def run(self, task: str, **kwargs) -> str:
        """Execute the strategy synchronously."""
        ...

    async def arun(self, task: str, **kwargs) -> str:
        """Execute the strategy asynchronously (default: wraps sync run)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(task, **kwargs))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"
