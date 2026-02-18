"""
Sequential Pipeline strategy.

Simple chain: each agent refines the previous agent's output.

Flow:
  [Agent 1: Draft] → [Agent 2: Review] → [Agent 3: Polish] → Final
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Strategy

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger("jadeagent.council.pipeline")


class Pipeline(Strategy):
    """
    Sequential agent pipeline — each agent refines the previous output.

    Example:
        drafter = Agent(backend=groq, name="drafter",
            system_prompt="Write a first draft.")
        reviewer = Agent(backend=groq, name="reviewer",
            system_prompt="Review and improve this text.")
        polisher = Agent(backend=groq, name="polisher",
            system_prompt="Polish the final version.")

        pipeline = Pipeline([drafter, reviewer, polisher])
        result = pipeline.run("Write an essay about AI")
    """

    def __init__(
        self,
        agents: list[Agent],
        pass_template: str = "Previous agent's output:\n\n{previous}\n\nOriginal task: {task}\n\nPlease improve upon the previous output.",
        verbose: bool = True,
    ):
        self.agents = agents
        self.pass_template = pass_template
        self.verbose = verbose

    def run(self, task: str, **kwargs) -> str:
        """Execute agents sequentially, each refining the previous output."""
        current_output = ""

        for i, agent in enumerate(self.agents):
            if self.verbose:
                print(f"\n🔗 Pipeline [{i+1}/{len(self.agents)}] → {agent.name}")

            if i == 0:
                # First agent gets the raw task
                prompt = task
            else:
                # Subsequent agents get previous output + task
                prompt = self.pass_template.format(
                    previous=current_output,
                    task=task,
                )

            current_output = agent.chat(prompt)

            if self.verbose:
                preview = current_output[:150] + "..." if len(current_output) > 150 else current_output
                print(f"   📝 Output: {preview}")

        if self.verbose:
            print(f"\n✅ Pipeline complete ({len(self.agents)} agents)")

        return current_output

    @property
    def name(self) -> str:
        agent_names = " → ".join(a.name for a in self.agents)
        return f"Pipeline({agent_names})"
