"""
Tree of Thought + Validator Agent — arXiv 2024.

"Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent"

Architecture:
  [Reasoner 1] → Branch A, Branch B
  [Reasoner 2] → Branch C, Branch D
  [Reasoner 3] → Branch E, Branch F
           ↓
  [Validator] evaluates all branches → prunes bad ones → selects best

Key: Validator acts as a "thought filter" that discards faulty reasoning
paths, improving accuracy by +8.8pp on GSM8K over standard ToT.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Strategy

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger("jadeagent.council.tot")


class TreeOfThought(Strategy):
    """
    ToT with Validator: multiple reasoners explore branches,
    validator prunes and selects the best reasoning path.

    Example:
        r1 = Agent(groq, name="reasoner_1",
            system_prompt="Think step by step. Explore multiple approaches.")
        r2 = Agent(groq, name="reasoner_2",
            system_prompt="Think carefully. Consider edge cases.")
        r3 = Agent(groq, name="reasoner_3",
            system_prompt="Be creative in your reasoning approach.")

        validator = Agent(groq, name="validator",
            system_prompt="Evaluate reasoning paths for logical correctness.")

        tot = TreeOfThought(
            reasoners=[r1, r2, r3],
            validator=validator,
            branches_per_reasoner=2,
        )
        answer = tot.run("If a train leaves at 3pm going 60mph...")
    """

    def __init__(
        self,
        reasoners: list[Agent],
        validator: Agent,
        branches_per_reasoner: int = 2,
        verbose: bool = True,
    ):
        self.reasoners = reasoners
        self.validator = validator
        self.branches_per_reasoner = branches_per_reasoner
        self.verbose = verbose

    def run(self, task: str, **kwargs) -> str:
        """
        1. Each reasoner generates multiple reasoning branches
        2. Validator evaluates and prunes faulty branches
        3. Best branch is selected as final answer
        """
        all_branches: list[dict[str, str]] = []  # {reasoner, branch_id, reasoning}

        # Phase 1: Generate branches
        if self.verbose:
            print(f"\n🌳 Tree of Thought — Generating branches")

        for reasoner in self.reasoners:
            if self.verbose:
                print(f"\n  🧠 {reasoner.name} exploring {self.branches_per_reasoner} branches...")

            for branch_id in range(1, self.branches_per_reasoner + 1):
                reasoner.reset()
                prompt = (
                    f"Problem: {task}\n\n"
                    f"This is reasoning branch {branch_id}/{self.branches_per_reasoner}. "
                    f"{'Explore a different approach than your first instinct.' if branch_id > 1 else 'Think step by step.'}\n\n"
                    f"Show your complete reasoning process, then give your final answer."
                )

                reasoning = reasoner.chat(prompt)
                all_branches.append({
                    "reasoner": reasoner.name,
                    "branch_id": f"{reasoner.name}_branch_{branch_id}",
                    "reasoning": reasoning,
                })

                if self.verbose:
                    preview = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                    print(f"    🌿 Branch {branch_id}: {preview}")

        # Phase 2: Validate branches
        if self.verbose:
            print(f"\n🔍 Validator ({self.validator.name}) evaluating {len(all_branches)} branches...")

        branches_text = "\n\n" + "\n\n---\n\n".join(
            f"=== Branch: {b['branch_id']} (by {b['reasoner']}) ===\n{b['reasoning']}"
            for b in all_branches
        )

        self.validator.reset()
        validation = self.validator.chat(
            f"You are evaluating multiple reasoning paths for this problem:\n\n"
            f"Problem: {task}\n\n"
            f"Here are all the reasoning branches:\n{branches_text}\n\n"
            f"For each branch:\n"
            f"1. Check for logical errors, incorrect calculations, or flawed assumptions\n"
            f"2. Rate confidence (HIGH / MEDIUM / LOW)\n"
            f"3. Identify the strongest branch(es)\n\n"
            f"Then provide the BEST FINAL ANSWER by:\n"
            f"- Using the strongest reasoning path as the foundation\n"
            f"- Incorporating valid insights from other branches\n"
            f"- Correcting any errors found\n\n"
            f"Your final answer should be the most accurate and well-reasoned response."
        )

        if self.verbose:
            print(f"\n✅ ToT complete ({len(self.reasoners)} reasoners × {self.branches_per_reasoner} branches)")

        return validation

    @property
    def name(self) -> str:
        return f"ToT(reasoners={len(self.reasoners)}, branches={self.branches_per_reasoner})"
