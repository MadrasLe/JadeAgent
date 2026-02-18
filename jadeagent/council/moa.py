"""
Mixture of Agents (MoA) — ICLR 2025.

Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities"

Architecture:
  Layer 0:  [Proposer A] [Proposer B] [Proposer C]   (diverse proposals)
                 ↓              ↓              ↓
  Layer 1:  [Proposer A'] [Proposer B'] [Proposer C'] (refined with context)
                 ↓              ↓              ↓
  Final:             [Aggregator]                      (synthesizes best)

Key insight: LLMs produce better responses when given other models' outputs
as reference, even if those references are lower quality ("collaborativeness").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Strategy

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger("jadeagent.council.moa")


class MixtureOfAgents(Strategy):
    """
    MoA: Layered Proposers → Aggregator.

    Can mix local + cloud backends! E.g.:
    - Proposers: MegaGemm local (fast, free)
    - Aggregator: GPT-4o via API (high quality)

    Example:
        from jadeagent import Agent
        from jadeagent.backends import OpenAICompatBackend

        groq1 = OpenAICompatBackend("llama-3.1-8b-instant", ...)
        groq2 = OpenAICompatBackend("gemma2-9b-it", ...)
        groq3 = OpenAICompatBackend("mixtral-8x7b-32768", ...)

        moa = MixtureOfAgents(
            proposers=[
                Agent(groq1, name="llama_proposer"),
                Agent(groq2, name="gemma_proposer"),
                Agent(groq3, name="mixtral_proposer"),
            ],
            aggregator=Agent(groq1, name="aggregator",
                system_prompt="Synthesize the best answer from multiple proposals."),
            num_layers=2,
        )
        answer = moa.run("Explain quantum entanglement")
    """

    def __init__(
        self,
        proposers: list[Agent],
        aggregator: Agent,
        num_layers: int = 2,
        verbose: bool = True,
    ):
        self.proposers = proposers
        self.aggregator = aggregator
        self.num_layers = num_layers
        self.verbose = verbose

    def run(self, task: str, **kwargs) -> str:
        """
        Execute MoA: multiple layers of proposals → aggregation.

        Each layer's proposers receive the previous layer's outputs as context.
        Final aggregator synthesizes all proposals into one answer.
        """
        previous_outputs: list[str] = []

        for layer in range(self.num_layers):
            if self.verbose:
                print(f"\n🔄 MoA Layer {layer + 1}/{self.num_layers}")

            current_outputs = []
            for i, proposer in enumerate(self.proposers):
                # Build prompt with context from previous layer
                if previous_outputs:
                    context = "\n\n---\n\n".join(
                        f"Reference {j+1}:\n{out}"
                        for j, out in enumerate(previous_outputs)
                    )
                    prompt = (
                        f"Task: {task}\n\n"
                        f"Here are responses from other models for reference:\n\n"
                        f"{context}\n\n"
                        f"Using these as reference (but not copying them), "
                        f"provide your own improved response."
                    )
                else:
                    prompt = task

                if self.verbose:
                    print(f"  🧠 Proposer {i+1}/{len(self.proposers)}: {proposer.name}")

                # Reset proposer session for clean generation
                proposer.reset()
                output = proposer.chat(prompt)
                current_outputs.append(output)

                if self.verbose:
                    preview = output[:100] + "..." if len(output) > 100 else output
                    print(f"     └─ {preview}")

            previous_outputs = current_outputs

        # Final aggregation
        if self.verbose:
            print(f"\n🎯 Aggregating with: {self.aggregator.name}")

        proposals = "\n\n---\n\n".join(
            f"Proposal {i+1} (from {self.proposers[i].name}):\n{out}"
            for i, out in enumerate(previous_outputs)
        )

        self.aggregator.reset()
        final = self.aggregator.chat(
            f"You are given multiple expert proposals for the following task.\n\n"
            f"Task: {task}\n\n"
            f"Proposals:\n{proposals}\n\n"
            f"Synthesize the best possible answer by combining the strongest "
            f"points from each proposal. Be comprehensive and accurate."
        )

        if self.verbose:
            print(f"\n✅ MoA complete ({self.num_layers} layers × {len(self.proposers)} proposers)")

        return final

    @property
    def name(self) -> str:
        return f"MoA(proposers={len(self.proposers)}, layers={self.num_layers})"
