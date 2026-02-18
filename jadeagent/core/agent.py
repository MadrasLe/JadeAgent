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
        skill_library=None,
        skill_generator=None,
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

        # SkillLibrary + SkillGenerator (Voyager-style)
        self.skill_library = skill_library
        self.skill_generator = skill_generator

        # Load existing skills from library into registry
        if self.skill_library:
            for skill_tool in self.skill_library.all_tools():
                self.tools.register(skill_tool)
            if self.verbose and len(self.skill_library) > 0:
                print(f"  Loaded {len(self.skill_library)} skills from library")

        # Register meta-tool for self-improving skills
        if self.skill_generator:
            self._register_meta_tool()
            # Enhance system prompt
            self._system_prompt += (
                "\n\nYou have a special tool called 'create_and_use_tool'. "
                "When you need to perform an action but don't have the right tool, "
                "call create_and_use_tool with a description of what the tool should do "
                "and the arguments you want to pass. A new tool will be generated and "
                "executed for you automatically."
            )

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
                        print(f"  > Calling tool: {tc.name}({tc.arguments})")

                    # Auto-skill: if tool not found, try to create it
                    if tc.name not in self.tools._tools:
                        self._try_auto_skill(tc.name, task)

                    # Execute tool
                    result = self.tools.execute(tc)
                    tool_calls_made.append(tc)

                    if self.verbose:
                        result_preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"  <- Result: {result_preview}")

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

    def _try_auto_skill(self, tool_name: str, task: str):
        """Try to find or generate a missing tool."""
        # 1. Search SkillLibrary
        if self.skill_library:
            matches = self.skill_library.search(tool_name)
            if matches:
                self.tools.register(matches[0])
                if self.verbose:
                    print(f"  [SKILL] Found in library: {matches[0].name}")
                return

        # 2. Generate with SkillGenerator
        if self.skill_generator:
            if self.verbose:
                print(f"  [SKILL] Generating new skill: {tool_name}...")

            new_tool = self.skill_generator.generate(
                task=f"Create a function called '{tool_name}' that helps with: {task}",
                context=f"An AI agent needs this tool to complete its task.",
            )

            if new_tool:
                self.tools.register(new_tool)

                # Save to library for future reuse
                if self.skill_library and self.skill_generator.last_code:
                    self.skill_library.save(
                        name=new_tool.name,
                        code=self.skill_generator.last_code,
                        description=new_tool.description,
                        overwrite=True,
                    )
                    if self.verbose:
                        print(f"  [SKILL] Saved to library: {new_tool.name}")
            else:
                if self.verbose:
                    print(f"  [SKILL] Failed to generate: {tool_name}")

    def _register_meta_tool(self):
        """Register the create_and_use_tool meta-tool for self-improving agents."""
        agent_ref = self  # Closure reference

        def create_and_use_tool(task_description: str, input_text: str = "") -> str:
            """Create a new tool on-the-fly and use it immediately.

            Args:
                task_description: What the tool should do (e.g. 'reverse a string').
                input_text: The input to pass to the newly created tool.

            Returns:
                The result of running the new tool.
            """
            if agent_ref.verbose:
                print(f"  [SKILL] Generating tool for: {task_description}")

            new_tool = agent_ref.skill_generator.generate(
                task=task_description,
                context=f"Input to process: {input_text}" if input_text else "",
                skill_library=agent_ref.skill_library,  # Save happens inside generate()
            )

            if new_tool is None:
                return f"Failed to generate tool for: {task_description}"

            # Register for future use in this session
            agent_ref.tools.register(new_tool)

            # Execute immediately with input
            if input_text:
                try:
                    # Try to call with 'text' param (common pattern)
                    import inspect
                    sig = inspect.signature(new_tool.func)
                    params = list(sig.parameters.keys())
                    if params:
                        result = new_tool.func(**{params[0]: input_text})
                        return str(result)
                except Exception as e:
                    return f"Tool created ({new_tool.name}) but execution failed: {e}"

            return f"Tool '{new_tool.name}' created successfully. You can now call it directly."

        meta_tool = Tool(
            func=create_and_use_tool,
            name="create_and_use_tool",
            description="Create a new tool on-the-fly and optionally use it immediately. "
                       "Use this when you need a capability you don't have.",
        )
        self.tools.register(meta_tool)

    def __repr__(self) -> str:
        tools_str = f", tools={self.tools.names}" if self.tools else ""
        return f"<Agent({self.name}, backend={self.backend.name}{tools_str})>"
