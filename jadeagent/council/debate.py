"""
Multi-Agent Debate (MAD) — ACL 2024.

Liang et al., "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate"

Architecture:
  Round 1:  [Debater A argues]  [Debater B argues]
  Round 2:  [A sees B, refutes] [B sees A, refutes]
  Round 3:  [A final argument]  [B final argument]
  Final:    [Judge evaluates all arguments → verdict]

Solves "Degeneration-of-Thought" problem in self-reflection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Strategy

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger("jadeagent.council.debate")


class Debate(Strategy):
    """
    Multi-Agent Debate: agents argue opposing viewpoints, judge decides.

    Example:
        debater_a = Agent(groq1, name="optimist",
            system_prompt="You argue FOR the given position. Be persuasive.")
        debater_b = Agent(groq2, name="skeptic",
            system_prompt="You argue AGAINST the given position. Be critical.")
        judge = Agent(groq3, name="judge",
            system_prompt="Evaluate arguments fairly. Pick the strongest position.")

        debate = Debate(
            debaters=[debater_a, debater_b],
            judge=judge,
            rounds=3,
        )
        verdict = debate.run("Should AI development be regulated?")
    """

    def __init__(
        self,
        debaters: list[Agent],
        judge: Agent,
        rounds: int = 3,
        verbose: bool = True,
    ):
        if len(debaters) < 2:
            raise ValueError("Debate requires at least 2 debaters")
        self.debaters = debaters
        self.judge = judge
        self.rounds = rounds
        self.verbose = verbose

    def run(self, task: str, **kwargs) -> str:
        """
        Run a multi-round debate, then have the judge synthesize.

        Each round, debaters see previous arguments and refine their positions.
        """
        # Track all arguments per debater
        argument_history: dict[str, list[str]] = {
            d.name: [] for d in self.debaters
        }

        for round_num in range(1, self.rounds + 1):
            if self.verbose:
                print(f"\n⚔️  Debate Round {round_num}/{self.rounds}")

            round_arguments = {}

            for debater in self.debaters:
                # Build context from other debaters' previous arguments
                other_args = []
                for other_name, args in argument_history.items():
                    if other_name != debater.name and args:
                        latest = args[-1]
                        other_args.append(f"{other_name}'s last argument:\n{latest}")

                if round_num == 1:
                    prompt = (
                        f"Topic for debate: {task}\n\n"
                        f"Present your opening argument."
                    )
                else:
                    context = "\n\n---\n\n".join(other_args) if other_args else "No previous arguments."
                    prompt = (
                        f"Topic: {task}\n\n"
                        f"Round {round_num}: Here are the other debaters' arguments:\n\n"
                        f"{context}\n\n"
                        f"Respond to their points and strengthen your position. "
                        f"Address their strongest arguments directly."
                    )

                debater.reset()
                argument = debater.chat(prompt)
                round_arguments[debater.name] = argument
                argument_history[debater.name].append(argument)

                if self.verbose:
                    preview = argument[:120] + "..." if len(argument) > 120 else argument
                    print(f"  💬 {debater.name}: {preview}")

        # Judge evaluates
        if self.verbose:
            print(f"\n⚖️  Judge ({self.judge.name}) evaluating...")

        all_arguments = []
        for debater in self.debaters:
            args = argument_history[debater.name]
            formatted = "\n\n".join(
                f"Round {i+1}: {arg}" for i, arg in enumerate(args)
            )
            all_arguments.append(f"=== {debater.name} ===\n{formatted}")

        debate_transcript = "\n\n" + "\n\n---\n\n".join(all_arguments)

        self.judge.reset()
        verdict = self.judge.chat(
            f"You are judging a debate on the topic: {task}\n\n"
            f"Here is the full transcript:\n{debate_transcript}\n\n"
            f"Evaluate the arguments from all sides. Consider:\n"
            f"1. Strength of evidence and reasoning\n"
            f"2. How well each debater addressed opposing points\n"
            f"3. Overall persuasiveness\n\n"
            f"Provide your verdict: which position is strongest and why? "
            f"Then synthesize the best answer incorporating insights from all sides."
        )

        if self.verbose:
            print(f"\n✅ Debate complete ({self.rounds} rounds × {len(self.debaters)} debaters)")

        return verdict

    @property
    def name(self) -> str:
        names = ", ".join(d.name for d in self.debaters)
        return f"Debate({names}, judge={self.judge.name})"
