"""
Agent with ReAct (Reason + Act) loop and executable policy.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Iterator

from .session import Session
from .tools import Tool, ToolRegistry
from .types import AgentEvent, AgentResult, ToolCall
from ..backends.base import LLMBackend
from ..governance import NodeManifest, PolicyBundle, TaskPolicy
from ..state.events import JadeStateEvent
from ..state.manifest import JadeStateManifest, canonical_json_hash
from ..state.snapshot import AgentRuntimeSnapshot, tool_call_to_dict
from ..state.store import StateStore

logger = logging.getLogger("jadeagent.core.agent")


class Agent:
    """Autonomous agent with ReAct loop and tool calling."""

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
        policy_bundle: PolicyBundle | None = None,
        node_manifest: NodeManifest | None = None,
        audit_sink: Any = None,
        state_store: StateStore | None = None,
        run_id: str | None = None,
    ):
        self.backend = backend
        self.name = name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.audit_sink = audit_sink
        self.state_store = state_store
        self.run_id = run_id
        self._configured_run_id = run_id
        self._state_manifest: JadeStateManifest | None = None

        self._system_prompt = system_prompt or (
            f"You are {name}, a helpful and intelligent AI assistant. "
            "Be direct and concise in your responses."
        )

        self.tools = ToolRegistry(tools)
        self.skill_library = skill_library
        self.skill_generator = skill_generator

        if self.skill_library is not None:
            for skill_tool in self.skill_library.all_tools():
                self.tools.register(skill_tool)
            if self.verbose and len(self.skill_library) > 0:
                print(f"  Loaded {len(self.skill_library)} skills from library")

        if self.skill_generator:
            self._register_meta_tool()
            self._system_prompt += (
                "\n\nYou have a special tool called 'create_and_use_tool'. "
                "When you need to perform an action but don't have the right tool, "
                "call create_and_use_tool with a description of what the tool should do "
                "and the arguments you want to pass. A new tool will be generated and "
                "executed for you automatically."
            )

        self.node_manifest = node_manifest or NodeManifest(
            node_id=name,
            role="agent",
            capabilities=tuple(self.tools.names),
            constitution=policy_bundle or PolicyBundle(),
        )
        if policy_bundle is not None:
            self.node_manifest = replace(self.node_manifest, constitution=policy_bundle)
        if not self.node_manifest.capabilities:
            self.node_manifest = replace(
                self.node_manifest,
                capabilities=tuple(self.tools.names),
            )

        self._active_task_policy: TaskPolicy | None = None
        self._active_execution_context: dict[str, Any] = {}
        self.session = Session(backend, system_prompt=self._system_prompt)

    def run(
        self,
        task: str,
        task_policy: TaskPolicy | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        previous_task_policy = self._active_task_policy
        previous_context = self._active_execution_context
        self._active_task_policy = task_policy
        self._active_execution_context = dict(task_context or {})
        try:
            self._start_state_run(task, task_policy=task_policy, task_context=task_context)
            events: list[AgentEvent] = []
            tool_calls_made: list[ToolCall] = []

            if self.verbose:
                print(f"\n[Agent {self.name}] Starting task: {task[:80]}...")

            start_time = time.time()
            self._checkpoint_state("PLANNING", step=0, metadata={"task": task})

            for step in range(1, self.max_iterations + 1):
                if self.verbose:
                    print(f"\n--- Step {step}/{self.max_iterations} ---")

                self._checkpoint_state("AWAITING_MODEL", step=step)
                response = self.session.chat(
                    task if step == 1 else "Continue based on the tool results above.",
                    tools=self.tools.schemas if self.tools else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self._checkpoint_state(
                    "OBSERVING",
                    step=step,
                    metadata={
                        "finish_reason": response.finish_reason,
                        "model": response.model,
                        "has_tool_calls": response.has_tool_calls,
                        **self._response_usage_metadata(response),
                    },
                )

                if response.has_tool_calls:
                    for tc in response.tool_calls:
                        if self.verbose:
                            print(f"  > Calling tool: {tc.name}({tc.arguments})")

                        if tc.name not in self.tools._tools:
                            self._try_auto_skill(tc.name, task)

                        self._checkpoint_state("READY_TOOL", step=step, pending_tool_call=tc)
                        result = self._execute_tool_call_idempotent(tc, step)
                        tool_calls_made.append(tc)

                        if self.verbose:
                            result_preview = result[:200] + "..." if len(result) > 200 else result
                            print(f"  <- Result: {result_preview}")

                        events.append(AgentEvent(type="tool_call", tool_call=tc, step=step))
                        events.append(AgentEvent(
                            type="tool_result",
                            tool_result=result,
                            content=tc.name,
                            step=step,
                        ))
                        self.session.add_tool_result(tc.id, tc.name, result)
                        self._checkpoint_state(
                            "OBSERVING",
                            step=step,
                            last_observation={"tool": tc.name, "result": result},
                        )

                    if step == self.max_iterations:
                        self._checkpoint_state("AWAITING_MODEL", step=step)
                        final_response = self.session.chat(
                            "Provide the final answer using the tool results above. Do not call any tools.",
                            tools=None,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        answer = final_response.content or ""

                        if self.verbose:
                            elapsed = time.time() - start_time
                            print(f"\n[Agent {self.name}] Done in {step} steps ({elapsed:.1f}s)")
                            print(f"Answer: {answer[:200]}...")

                        events.append(AgentEvent(type="answer", content=answer, step=step))
                        self._checkpoint_state(
                            "COMPLETED",
                            step=step,
                            metadata={"answer": answer, **self._response_usage_metadata(final_response)},
                        )
                        self._emit_state_event("run_completed", phase="COMPLETED", step=step)
                        return AgentResult(
                            answer=answer,
                            steps=step,
                            tool_calls_made=tool_calls_made,
                            events=events,
                            usage=final_response.usage or response.usage,
                        )

                else:
                    answer = response.content or ""
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"\n[Agent {self.name}] Done in {step} steps ({elapsed:.1f}s)")
                        print(f"Answer: {answer[:200]}...")

                    events.append(AgentEvent(type="answer", content=answer, step=step))
                    self._checkpoint_state(
                        "COMPLETED",
                        step=step,
                        metadata={"answer": answer, **self._response_usage_metadata(response)},
                    )
                    self._emit_state_event("run_completed", phase="COMPLETED", step=step)
                    return AgentResult(
                        answer=answer,
                        steps=step,
                        tool_calls_made=tool_calls_made,
                        events=events,
                        usage=response.usage,
                    )

            last_content = self.session.messages[-1].content or ""
            logger.warning(f"Agent {self.name} hit max iterations ({self.max_iterations})")
            events.append(AgentEvent(
                type="error",
                content=f"Max iterations ({self.max_iterations}) reached.",
                step=self.max_iterations,
            ))
            self._checkpoint_state(
                "FAILED",
                step=self.max_iterations,
                metadata={"error": f"Max iterations ({self.max_iterations}) reached."},
            )
            self._emit_state_event("run_failed", phase="FAILED", step=self.max_iterations)
            return AgentResult(
                answer=last_content,
                steps=self.max_iterations,
                tool_calls_made=tool_calls_made,
                events=events,
            )
        except Exception as exc:
            self._checkpoint_state("FAILED", step=locals().get("step", 0), metadata={"error": repr(exc)})
            self._emit_state_event("run_failed", phase="FAILED", step=locals().get("step", 0), message=repr(exc))
            raise
        finally:
            self._active_task_policy = previous_task_policy
            self._active_execution_context = previous_context

    def chat(
        self,
        message: str,
        task_policy: TaskPolicy | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> str:
        previous_task_policy = self._active_task_policy
        previous_context = self._active_execution_context
        self._active_task_policy = task_policy
        self._active_execution_context = dict(task_context or {})
        try:
            response = self.session.chat(
                message,
                tools=self.tools.schemas if self.tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                for tc in response.tool_calls:
                    result = self._execute_tool_call(tc)
                    self.session.add_tool_result(tc.id, tc.name, result)

                response = self.session.chat(
                    "Respond based on the tool results.",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

            return response.content or ""
        finally:
            self._active_task_policy = previous_task_policy
            self._active_execution_context = previous_context

    def stream_run(
        self,
        task: str,
        task_policy: TaskPolicy | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> Iterator[AgentEvent]:
        previous_task_policy = self._active_task_policy
        previous_context = self._active_execution_context
        self._active_task_policy = task_policy
        self._active_execution_context = dict(task_context or {})
        try:
            yield AgentEvent(type="thinking", content=f"Starting: {task}", step=0)

            for step in range(1, self.max_iterations + 1):
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
                            result = self._execute_tool_call(tc)
                            yield AgentEvent(
                                type="tool_result",
                                tool_result=result,
                                content=tc.name,
                                step=step,
                            )
                            self.session.add_tool_result(tc.id, tc.name, result)
                        break
                else:
                    answer = "".join(full_content)
                    yield AgentEvent(type="answer", content=answer, step=step)
                    return

            yield AgentEvent(
                type="error",
                content=f"Max iterations ({self.max_iterations}) reached.",
                step=self.max_iterations,
            )
        finally:
            self._active_task_policy = previous_task_policy
            self._active_execution_context = previous_context

    def reset(self):
        self.session.reset()

    def save_state(
        self,
        phase: str = "MANUAL",
        *,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRuntimeSnapshot | None:
        """Manually checkpoint the agent session into the configured StateStore."""

        return self._checkpoint_state(phase, step=step, metadata=metadata)

    def restore_state(
        self,
        snapshot_or_run_id: AgentRuntimeSnapshot | dict | str,
        *,
        state_store: StateStore | None = None,
        task_policy: TaskPolicy | None = None,
        validate: bool = True,
    ) -> AgentRuntimeSnapshot:
        """Restore session state from a snapshot or from the latest snapshot of a run."""

        store = state_store or self.state_store
        manifest = None
        if isinstance(snapshot_or_run_id, str):
            if store is None:
                raise ValueError("restore_state(run_id) requires a StateStore")
            capsule = store.load_run(snapshot_or_run_id)
            manifest = capsule.manifest
            snapshot = capsule.latest_snapshot
            if snapshot is None:
                raise ValueError(f"run {snapshot_or_run_id!r} has no snapshots")
            self.run_id = snapshot_or_run_id
            self._state_manifest = manifest
        else:
            snapshot = (
                snapshot_or_run_id
                if isinstance(snapshot_or_run_id, AgentRuntimeSnapshot)
                else AgentRuntimeSnapshot.from_dict(snapshot_or_run_id)
            )

        if validate and manifest is not None:
            from ..state.compatibility import validate_restore_compatibility

            validate_restore_compatibility(
                manifest,
                tenant_id=self.node_manifest.tenant_id or None,
                policy_hash=self._policy_hash(task_policy) or None,
                tool_registry_hash=self._tool_registry_hash() or None,
                model_fingerprint=self._backend_fingerprint() or None,
                backend=self.backend.name,
            ).require_allowed()

        self.session.restore_snapshot(snapshot.session)
        return snapshot

    def _start_state_run(
        self,
        task: str,
        *,
        task_policy: TaskPolicy | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> JadeStateManifest | None:
        if self.state_store is None:
            return None

        context = dict(task_context or {})
        run_id = str(context.get("run_id") or self._configured_run_id or "")
        manifest = JadeStateManifest(
            run_id=run_id or uuid.uuid4().hex,
            task_id=str(context.get("task_id", "")),
            agent_id=self.name,
            tenant_id=str(context.get("tenant_id") or self.node_manifest.tenant_id or ""),
            capability=str(context.get("capability") or ",".join(self.tools.names)),
            backend=self.backend.name,
            model_fingerprint=self._backend_fingerprint(),
            policy_hash=self._policy_hash(task_policy),
            tool_registry_hash=self._tool_registry_hash(),
            memory_scope_hash=self._memory_scope_hash(context),
            metadata={
                "task_preview": task[:240],
                "node_manifest": self.node_manifest.to_dict(),
            },
        )
        self.state_store.create_run(manifest)
        self.run_id = manifest.run_id
        self._state_manifest = manifest
        self._emit_state_event(
            "run_started",
            phase="NEW",
            step=0,
            message="agent run started",
            payload={"task": task},
        )
        return manifest

    def _checkpoint_state(
        self,
        phase: str,
        *,
        step: int = 0,
        pending_tool_call: ToolCall | None = None,
        last_observation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRuntimeSnapshot | None:
        if self.state_store is None or not self.run_id:
            return None

        snapshot = AgentRuntimeSnapshot(
            phase=phase,
            step=step,
            session=self.session.snapshot(metadata={"agent": self.name}),
            pending_tool_call=tool_call_to_dict(pending_tool_call),
            last_observation=dict(last_observation or {}),
            memory_refs=[
                mount.name
                for mount in self.node_manifest.memory_mounts
            ],
            metadata=dict(metadata or {}),
        )
        self.state_store.save_snapshot(self.run_id, snapshot)
        self._emit_state_event(
            "checkpoint",
            phase=phase,
            step=step,
            message="state checkpoint saved",
            payload={"snapshot_id": snapshot.snapshot_id},
        )
        return snapshot

    def _emit_state_event(
        self,
        event_type: str,
        *,
        phase: str = "",
        step: int = 0,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> JadeStateEvent | None:
        if self.state_store is None or not self.run_id:
            return None
        event = JadeStateEvent(
            event_type=event_type,
            run_id=self.run_id,
            phase=phase,
            step=step,
            message=message,
            actor=self.name,
            payload=dict(payload or {}),
        )
        return self.state_store.append_event(self.run_id, event)

    def _response_usage_metadata(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
        }

    def _policy_hash(self, task_policy: TaskPolicy | None = None) -> str:
        return canonical_json_hash({
            "node_constitution": self.node_manifest.constitution.to_dict(),
            "task_policy": task_policy.to_dict() if task_policy is not None else {},
        })

    def _tool_registry_hash(self) -> str:
        tools = []
        for name in sorted(self.tools._tools):
            tool_obj = self.tools._tools[name]
            tools.append({
                "name": tool_obj.name,
                "schema": tool_obj.schema.to_dict(),
                "effects": list(tool_obj.effects),
                "resource_refs": list(tool_obj.resource_refs),
                "read_path_args": list(tool_obj.read_path_args),
                "write_path_args": list(tool_obj.write_path_args),
                "metadata": dict(tool_obj.metadata),
            })
        return canonical_json_hash(tools)

    def _backend_fingerprint(self) -> str:
        model_name = (
            getattr(self.backend, "model", "")
            or getattr(self.backend, "model_name", "")
            or getattr(self.backend, "model_id", "")
        )
        return canonical_json_hash({
            "backend": self.backend.name,
            "model": str(model_name),
            "supports_kv_persistence": self.backend.supports_kv_persistence,
            "supports_tool_calling": self.backend.supports_tool_calling,
        })

    def _memory_scope_hash(self, context: dict[str, Any] | None = None) -> str:
        context = dict(context or {})
        return canonical_json_hash({
            "tenant_id": context.get("tenant_id") or self.node_manifest.tenant_id,
            "memory_scope": context.get("memory_scope", ""),
            "mounts": [mount.to_dict() for mount in self.node_manifest.memory_mounts],
        })

    def _tool_idempotency_key(self, tool_call: ToolCall, step: int) -> str:
        return canonical_json_hash({
            "kind": "tool_call",
            "run_id": self.run_id or "",
            "step": int(step),
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        })

    def _lookup_tool_result(self, idempotency_key: str) -> str | None:
        if self.state_store is None or not self.run_id or not idempotency_key:
            return None
        try:
            events = self.state_store.list_events(self.run_id, limit=10_000)
        except Exception:
            logger.debug("Failed to read state events for tool replay", exc_info=True)
            return None

        for event in reversed(events):
            if event.event_type != "tool_result_recorded":
                continue
            payload = event.payload or {}
            if payload.get("idempotency_key") == idempotency_key:
                return str(payload.get("result", ""))
        return None

    def _record_tool_result(
        self,
        tool_call: ToolCall,
        *,
        step: int,
        result: str,
        idempotency_key: str,
        reused: bool = False,
    ) -> None:
        event_type = "tool_result_reused" if reused else "tool_result_recorded"
        self._emit_state_event(
            event_type,
            phase="TOOL_RESULT",
            step=step,
            message=(
                "tool result reused from idempotency record"
                if reused
                else "tool result recorded for idempotent replay"
            ),
            payload={
                "idempotency_key": idempotency_key,
                "tool_call": tool_call_to_dict(tool_call),
                "tool_name": tool_call.name,
                "result": result,
                "reused": reused,
            },
        )

    def _execute_tool_call_idempotent(self, tool_call: ToolCall, step: int) -> str:
        if self.state_store is None or not self.run_id:
            return self._execute_tool_call(tool_call)

        idempotency_key = self._tool_idempotency_key(tool_call, step)
        previous_result = self._lookup_tool_result(idempotency_key)
        if previous_result is not None:
            self._record_tool_result(
                tool_call,
                step=step,
                result=previous_result,
                idempotency_key=idempotency_key,
                reused=True,
            )
            return previous_result

        result = self._execute_tool_call(tool_call)
        self._record_tool_result(
            tool_call,
            step=step,
            result=result,
            idempotency_key=idempotency_key,
            reused=False,
        )
        return result

    def _execute_tool_call(self, tool_call: ToolCall) -> str:
        return self.tools.execute(
            tool_call,
            node_manifest=self.node_manifest,
            task_policy=self._active_task_policy,
            audit_sink=self.audit_sink,
            execution_context={
                "node_id": self.node_manifest.node_id,
                **self._active_execution_context,
            },
        )

    def _dynamic_tool_creation_allowed(self) -> bool:
        allowed = self.node_manifest.constitution.allow_dynamic_tool_creation
        if self._active_task_policy is not None:
            override = self._active_task_policy.allow_dynamic_tool_creation
            if override is not None:
                allowed = allowed and bool(override)
        return allowed

    def _try_auto_skill(self, tool_name: str, task: str):
        if self.skill_library is not None:
            matches = self.skill_library.search(tool_name)
            if matches:
                self.tools.register(matches[0])
                if self.verbose:
                    print(f"  [SKILL] Found in library: {matches[0].name}")
                return

        if not self._dynamic_tool_creation_allowed():
            if self.verbose:
                print(f"  [POLICY] Dynamic tool creation blocked for: {tool_name}")
            return

        if self.skill_generator:
            if self.verbose:
                print(f"  [SKILL] Generating new skill: {tool_name}...")

            new_tool = self.skill_generator.generate(
                task=f"Create a function called '{tool_name}' that helps with: {task}",
                context=f"An AI agent needs this tool to complete its task.",
            )

            if new_tool:
                self.tools.register(new_tool)

                if self.skill_library is not None and self.skill_generator.last_code:
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
        agent_ref = self

        def create_and_use_tool(task_description: str, input_text: str = "") -> str:
            if agent_ref.verbose:
                print(f"  [SKILL] Generating tool for: {task_description}")

            new_tool = agent_ref.skill_generator.generate(
                task=task_description,
                context=f"Input to process: {input_text}" if input_text else "",
                skill_library=agent_ref.skill_library,
            )

            if new_tool is None:
                return f"Failed to generate tool for: {task_description}"

            agent_ref.tools.register(new_tool)

            if input_text:
                try:
                    import inspect

                    sig = inspect.signature(new_tool.func)
                    params = list(sig.parameters.keys())
                    if params:
                        return new_tool.execute(
                            {params[0]: input_text},
                            node_manifest=agent_ref.node_manifest,
                            task_policy=agent_ref._active_task_policy,
                            audit_sink=agent_ref.audit_sink,
                            execution_context=agent_ref._active_execution_context,
                        )
                except Exception as e:
                    return f"Tool created ({new_tool.name}) but execution failed: {e}"

            return f"Tool '{new_tool.name}' created successfully. You can now call it directly."

        meta_tool = Tool(
            func=create_and_use_tool,
            name="create_and_use_tool",
            description=(
                "Create a new tool on-the-fly and optionally use it immediately. "
                "Use this when you need a capability you don't have."
            ),
            effects=["generate_tool"],
            resource_refs=["tool.execute:create_and_use_tool"],
        )
        self.tools.register(meta_tool)

    def __repr__(self) -> str:
        tools_str = f", tools={self.tools.names}" if self.tools else ""
        return f"<Agent({self.name}, backend={self.backend.name}{tools_str})>"
