"""
Agent with ReAct (Reason + Act) loop.

Implements the Plan → Act → Reflect cycle:
1. Agent receives a task
2. Generates a response (may include tool calls)
3. Executes tool calls and feeds results back
4. Repeats until a final answer is produced

Evolved from JADE's cognitive loop (Remember → See → Respond → Memorize)
with tool calling support and multi-backend compatibility.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator

from .session import Session
from .tools import Tool, ToolRegistry
from .types import AgentEvent, AgentResult, Message, Response, ToolCall
from ..backends.base import LLMBackend

logger = logging.getLogger("jadeagent.core.agent")


class Agent:
    """
    Autonomous agent with ReAct loop and tool calling.

    Example:
        from jadeagent import Agent, tool
        from jadeagent.backends import OpenAICompatBackend

        @tool(description="Calculate a math expression")
        def calculator(expression: str) -> str:
            return str(eval(expression))

        backend = OpenAICompatBackend("gpt-4o", api_key="sk-...")
        agent = Agent(
            backend=backend,
            name="math_tutor",
            system_prompt="You are a helpful math tutor.",
            tools=[calculator],
        )

        result = agent.run("What is 15% of 340?")
        print(result.answer)
    """

    def __init__(
        self,
        backend: LLMBackend,
        name: str = "jade",
        system_prompt: str | None = None,
        tools: list[Tool | Callable] | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        verbose: bool = True,
    ):
        self.backend = backend
        self.name = name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Build system prompt
        self._system_prompt = system_prompt or (
            f"You are {name}, a helpful and intelligent AI assistant. "
            "Be direct and concise in your responses."
        )

        # Tool registry
        self.tools = ToolRegistry(tools)

        # Session for multi-turn
        self.session = Session(backend, system_prompt=self._system_prompt)

    def run(self, task: str) -> AgentResult:
        """
        Execute the ReAct loop to solve a task.

        The agent will:
        1. Think about the task
        2. Call tools if needed
        3. Observe tool results
        4. Repeat until it produces a final answer

        Args:
            task: The task or question for the agent.

        Returns:
            AgentResult with the answer, steps taken, and events.
        """
        events: list[AgentEvent] = []
        tool_calls_made: list[ToolCall] = []

        if self.verbose:
            print(f"\n🤖 [{self.name}] Starting task: {task[:80]}...")

        start_time = time.time()

        for step in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Step {step}/{self.max_iterations} ---")

            # Generate response
            response = self.session.chat(
                task if step == 1 else "Continue based on the tool results above.",
                tools=self.tools.schemas if self.tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Check for tool calls
            if response.has_tool_calls:
                for tc in response.tool_calls:
                    if self.verbose:
                        print(f"  🔧 Calling tool: {tc.name}({tc.arguments})")

                    # Execute tool
                    result = self.tools.execute(tc)
                    tool_calls_made.append(tc)

                    if self.verbose:
                        result_preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"  📋 Result: {result_preview}")

                    # Record events
                    events.append(AgentEvent(
                        type="tool_call", tool_call=tc, step=step,
                    ))
                    events.append(AgentEvent(
                        type="tool_result", tool_result=result,
                        content=tc.name, step=step,
                    ))

                    # Feed result back to session
                    self.session.add_tool_result(tc.id, tc.name, result)

            else:
                # No tool calls → this is the final answer
                answer = response.content or ""

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"\n✅ [{self.name}] Done in {step} steps ({elapsed:.1f}s)")
                    print(f"💬 Answer: {answer[:200]}...")

                events.append(AgentEvent(
                    type="answer", content=answer, step=step,
                ))

                return AgentResult(
                    answer=answer,
                    steps=step,
                    tool_calls_made=tool_calls_made,
                    events=events,
                    usage=response.usage,
                )

        # Max iterations reached
        last_content = self.session.messages[-1].content or ""
        logger.warning(f"Agent {self.name} hit max iterations ({self.max_iterations})")

        events.append(AgentEvent(
            type="error",
            content=f"Max iterations ({self.max_iterations}) reached.",
            step=self.max_iterations,
        ))

        return AgentResult(
            answer=last_content,
            steps=self.max_iterations,
            tool_calls_made=tool_calls_made,
            events=events,
        )

    def chat(self, message: str) -> str:
        """
        Simple chat without tool loop (single turn).

        For multi-step tasks with tools, use run() instead.

        Args:
            message: User message.

        Returns:
            Assistant response text.
        """
        response = self.session.chat(
            message,
            tools=self.tools.schemas if self.tools else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # If there are tool calls in chat mode, execute them once
        if response.has_tool_calls:
            for tc in response.tool_calls:
                result = self.tools.execute(tc)
                self.session.add_tool_result(tc.id, tc.name, result)

            # Get final response after tool execution
            response = self.session.chat(
                "Respond based on the tool results.",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        return response.content or ""

    def stream_run(self, task: str) -> Iterator[AgentEvent]:
        """
        Execute the ReAct loop, yielding events for real-time UI.

        Yields AgentEvent objects with types:
        - "thinking": Agent is reasoning
        - "tool_call": Agent is calling a tool
        - "tool_result": Tool returned a result
        - "token": Streaming token
        - "answer": Final answer

        Args:
            task: The task or question.

        Yields:
            AgentEvent objects.
        """
        yield AgentEvent(type="thinking", content=f"Starting: {task}", step=0)

        for step in range(1, self.max_iterations + 1):
            # Stream response
            full_content = []
            for chunk in self.session.stream_chat(
                task if step == 1 else "Continue based on the tool results.",
                tools=self.tools.schemas if self.tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                if chunk.token:
                    full_content.append(chunk.token)
                    yield AgentEvent(type="token", content=chunk.token, step=step)

                if chunk.finished and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        yield AgentEvent(type="tool_call", tool_call=tc, step=step)

                        result = self.tools.execute(tc)
                        yield AgentEvent(
                            type="tool_result", tool_result=result,
                            content=tc.name, step=step,
                        )
                        self.session.add_tool_result(tc.id, tc.name, result)
                    break
            else:
                # No tool calls → final answer
                answer = "".join(full_content)
                yield AgentEvent(type="answer", content=answer, step=step)
                return

        yield AgentEvent(
            type="error",
            content=f"Max iterations ({self.max_iterations}) reached.",
            step=self.max_iterations,
        )

    def reset(self):
        """Reset the agent's conversation history."""
        self.session.reset()

    def __repr__(self) -> str:
        tools_str = f", tools={self.tools.names}" if self.tools else ""
        return f"<Agent({self.name}, backend={self.backend.name}{tools_str})>"
