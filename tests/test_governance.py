"""Runtime governance tests for strict node/task permissions."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, r"c:\Users\gabri\JadeAgent")

from jadeagent import Agent, FilesystemPolicy, NodeManifest, PolicyBundle, TaskPolicy, tool
from jadeagent.backends.base import LLMBackend
from jadeagent.core.types import Message, Response, StreamChunk, ToolCall


@tool(
    description="Write content to a file",
    effects=["write"],
    write_path_args=["path"],
)
def write_report(path: str, content: str) -> str:
    return f"WROTE {path}: {content}"


class FakeBackend(LLMBackend):
    def __init__(self, responses: list[Response]):
        self._responses = list(responses)

    def chat(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ) -> Response:
        if not self._responses:
            return Response(content="done")
        return self._responses.pop(0)

    def stream(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ):
        if False:
            yield StreamChunk()
        return


class GovernanceTests(unittest.TestCase):
    def test_read_only_task_blocks_write_effect(self):
        result = write_report.execute(
            {"path": "notes.txt", "content": "secret"},
            task_policy=TaskPolicy(read_only=True),
        )
        self.assertIn("Policy denied", result)
        self.assertIn("read-only", result)

    def test_write_root_allowlist_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_root = os.path.join(tmpdir, "allowed")
            blocked_root = os.path.join(tmpdir, "blocked")
            os.makedirs(allowed_root, exist_ok=True)
            os.makedirs(blocked_root, exist_ok=True)

            manifest = NodeManifest(
                node_id="writer_node",
                constitution=PolicyBundle(
                    filesystem=FilesystemPolicy(
                        allow_write_all=False,
                        allow_write_roots=(allowed_root,),
                    )
                ),
            )

            allowed_result = write_report.execute(
                {"path": os.path.join(allowed_root, "ok.txt"), "content": "x"},
                node_manifest=manifest,
            )
            blocked_result = write_report.execute(
                {"path": os.path.join(blocked_root, "no.txt"), "content": "x"},
                node_manifest=manifest,
            )

            self.assertIn("WROTE", allowed_result)
            self.assertIn("Policy denied", blocked_result)
            self.assertIn("outside allowed roots", blocked_result)

    def test_agent_run_applies_task_policy_during_tool_execution(self):
        backend = FakeBackend([
            Response(tool_calls=[
                ToolCall(id="call_1", name="write_report", arguments={
                    "path": "report.txt",
                    "content": "hello",
                })
            ]),
            Response(content="final answer"),
        ])

        agent = Agent(
            backend=backend,
            tools=[write_report],
            verbose=False,
        )
        result = agent.run("Write a report", task_policy=TaskPolicy(read_only=True))

        tool_results = [event.tool_result for event in result.events if event.type == "tool_result"]
        self.assertTrue(any(tool_result and "Policy denied" in tool_result for tool_result in tool_results))
        self.assertEqual(result.answer, "final answer")


if __name__ == "__main__":
    unittest.main()
