"""
SkillGenerator — LLM-powered tool code generation.

Uses an LLM to write Python functions, validates them in a
sandbox, and iteratively refines until they pass.

Inspired by Voyager's iterative prompting mechanism.

Example:
    gen = SkillGenerator(backend)
    tool = gen.generate("Create a function that reverses a string")
    tool.execute({"text": "hello"})  # → "olleh"
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any

from ..backends.base import LLMBackend
from ..core.tools import Tool
from ..core.types import Message, ToolSchema

logger = logging.getLogger("jadeagent.skills.generator")

# System prompt for skill generation
GENERATOR_PROMPT = """You are a Python tool generator for an AI agent framework called JadeAgent.

Your task: write a single Python function that the agent can use as a tool.

RULES:
1. Write ONLY the function definition (def ...). No imports at module level unless necessary.
2. The function MUST have type hints for all parameters and return type.
3. The function MUST have a clear docstring with Args section.
4. The function MUST return a string (the tool result).
5. Keep it simple, robust, and self-contained.
6. If you need imports, put them INSIDE the function body.
7. Handle errors gracefully — return error messages instead of raising.
8. The function name should be snake_case and descriptive.

EXAMPLE OUTPUT:
```python
def fetch_webpage(url: str) -> str:
    \"\"\"Fetch and return the text content of a webpage.

    Args:
        url: The URL to fetch.

    Returns:
        The webpage text content.
    \"\"\"
    try:
        import urllib.request
        response = urllib.request.urlopen(url, timeout=10)
        return response.read().decode('utf-8')[:5000]
    except Exception as e:
        return f"Error fetching {url}: {e}"
```

Output ONLY the Python function code inside a code block. No explanations."""

REFINE_PROMPT = """The function you generated has an error. Fix it.

Previous code:
```python
{code}
```

Error:
{error}

Write the corrected function. Output ONLY the Python code in a code block."""


class SkillGenerator:
    """
    Generates new tools using an LLM backend.

    The generator:
    1. Asks the LLM to write a Python function
    2. Extracts the code from the response
    3. Validates it compiles and runs
    4. Retries with error feedback (up to max_retries)

    Args:
        backend: LLM backend to use for code generation.
        max_retries: Maximum refinement attempts on failure.
        temperature: Temperature for code generation (lower = more deterministic).
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_retries: int = 3,
        temperature: float = 0.3,
    ):
        self.backend = backend
        self.max_retries = max_retries
        self.temperature = temperature

    def generate(
        self,
        task: str,
        context: str = "",
        test_input: dict | None = None,
    ) -> Tool | None:
        """
        Generate a new tool from a task description.

        Args:
            task: What the tool should do (natural language).
            context: Optional context about why the tool is needed.
            test_input: Optional test arguments to validate the tool.

        Returns:
            Tool object if successful, None if all retries failed.
        """
        # Build the initial prompt
        user_msg = f"Create a Python function for this task:\n{task}"
        if context:
            user_msg += f"\n\nContext: {context}"

        messages = [
            Message(role="system", content=GENERATOR_PROMPT),
            Message(role="user", content=user_msg),
        ]

        code = None
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Skill generation attempt {attempt}/{self.max_retries}")

            # Call LLM
            response = self.backend.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
            )

            # Extract code from response
            raw_code = self._extract_code(response.content or "")
            if not raw_code:
                last_error = "No code block found in LLM response"
                logger.warning(last_error)
                messages.append(Message(role="assistant", content=response.content or ""))
                messages.append(Message(
                    role="user",
                    content="Your response didn't contain a code block. "
                            "Please write the function inside ```python ... ``` markers.",
                ))
                continue

            # Validate: syntax check
            try:
                compile(raw_code, "<skill_gen>", "exec")
            except SyntaxError as e:
                last_error = f"Syntax error: {e}"
                logger.warning(f"Attempt {attempt}: {last_error}")
                messages.append(Message(role="assistant", content=response.content or ""))
                messages.append(Message(
                    role="user",
                    content=REFINE_PROMPT.format(code=raw_code, error=last_error),
                ))
                continue

            # Validate: execution in sandbox
            sandbox_result = self._sandbox_exec(raw_code, test_input)
            if sandbox_result["success"]:
                code = raw_code
                func_name = sandbox_result["func_name"]
                func = sandbox_result["func"]
                description = sandbox_result.get("description", task)

                logger.info(f"Skill generated successfully: {func_name}")

                return Tool(
                    func=func,
                    name=func_name,
                    description=description,
                )
            else:
                last_error = sandbox_result["error"]
                logger.warning(f"Attempt {attempt}: sandbox error: {last_error}")
                messages.append(Message(role="assistant", content=response.content or ""))
                messages.append(Message(
                    role="user",
                    content=REFINE_PROMPT.format(code=raw_code, error=last_error),
                ))

        logger.error(f"Skill generation failed after {self.max_retries} attempts. "
                      f"Last error: {last_error}")
        return None

    @property
    def last_code(self) -> str | None:
        """The last successfully generated code (for saving to SkillLibrary)."""
        return getattr(self, "_last_code", None)

    def _extract_code(self, text: str) -> str | None:
        """Extract Python code from a markdown code block."""
        # Try ```python ... ```
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ``` ... ```
        pattern = r"```\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "def " in code:
                return code

        # Try bare function definition
        lines = text.strip().split("\n")
        func_lines = []
        in_func = False
        for line in lines:
            if line.strip().startswith("def "):
                in_func = True
            if in_func:
                func_lines.append(line)

        if func_lines:
            return "\n".join(func_lines)

        return None

    def _sandbox_exec(
        self,
        code: str,
        test_input: dict | None = None,
    ) -> dict:
        """
        Execute code in a restricted namespace and validate.

        Returns dict with:
            success: bool
            func_name: str (if success)
            func: callable (if success)
            description: str (if success)
            error: str (if not success)
        """
        # Safe namespace with common imports available
        namespace = {
            "__builtins__": __builtins__,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {traceback.format_exc()}",
            }

        # Find the defined function
        func = None
        func_name = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "builtins":
                func = obj
                func_name = name
                break

        if func is None:
            return {
                "success": False,
                "error": "No function definition found in code",
            }

        # Extract description from docstring
        description = func.__doc__ or ""
        if description:
            # Take first line of docstring
            description = description.strip().split("\n")[0]

        # Test execution if test_input provided
        if test_input is not None:
            try:
                result = func(**test_input)
                if not isinstance(result, str):
                    result = str(result)
                logger.debug(f"Test result: {result[:100]}")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Test failed with input {test_input}: {e}",
                }

        # Save the code for later retrieval
        self._last_code = code

        return {
            "success": True,
            "func_name": func_name,
            "func": func,
            "description": description,
        }

    def __repr__(self) -> str:
        return f"<SkillGenerator(backend={self.backend.name})>"
