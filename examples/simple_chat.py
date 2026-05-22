"""
JadeAgent — Example: Simple Chat + Tool Agent

Demonstrates:
1. Using OpenAI-compatible backend (Groq, OpenAI, Ollama)
2. Using MegaGemm backend (local GPU)
3. Creating tools with @tool decorator
4. Running an agent with ReAct loop
"""

from jadeagent import Agent, tool
from jadeagent.backends import OpenAICompatBackend

# ─── Example 1: Simple Chat (Groq API) ──────────────────────────────────────

def example_simple_chat():
    """Basic chatbot with Groq API."""
    backend = OpenAICompatBackend(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key="YOUR_GROQ_KEY",  # Replace with your key
    )

    agent = Agent(
        backend=backend,
        name="jade",
        system_prompt="Você é J.A.D.E., uma assistente de IA calma e inteligente.",
    )

    # Simple multi-turn chat
    print(agent.chat("Olá! Como você está?"))
    print(agent.chat("Me explica o que é gravidade?"))
    print(agent.chat("E como isso se relaciona com a relatividade?"))


# ─── Example 2: Agent with Tools (Groq API) ────────────────────────────────

@tool(description="Calculate a mathematical expression")
def calculator(expression: str) -> str:
    """Evaluate a math expression safely.

    Args:
        expression: A mathematical expression to evaluate (e.g. '15 * 0.15')
    """
    try:
        # Safe eval — only math
        allowed = {"__builtins__": {}}
        import math
        allowed.update(vars(math))
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(description="Get the current date and time")
def get_datetime() -> str:
    """Return current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def example_tool_agent():
    """Agent with calculator and datetime tools."""
    backend = OpenAICompatBackend(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key="YOUR_GROQ_KEY",  # Replace with your key
    )

    agent = Agent(
        backend=backend,
        name="math_jade",
        system_prompt=(
            "You are a helpful math assistant. "
            "Use the calculator tool for any mathematical computation. "
            "Always show your work."
        ),
        tools=[calculator, get_datetime],
        verbose=True,
    )

    # This will trigger the ReAct loop:
    # 1. Agent thinks about the problem
    # 2. Calls calculator tool
    # 3. Uses result to formulate answer
    result = agent.run("What is 15% of 340, and what date is it today?")
    print(f"\nFinal Answer: {result.answer}")
    print(f"Steps taken: {result.steps}")
    print(f"Tool calls: {len(result.tool_calls_made)}")


# ─── Example 3: MegaGemm Local Backend ─────────────────────────────────────

def example_megagemm_local():
    """Agent running entirely on local GPU via MegaGemm."""
    from jadeagent.backends.megagemm import MegaGemmBackend

    backend = MegaGemmBackend(
        model="Qwen/Qwen2.5-7B-Instruct",
        quantize="int8",
        kv_offload=True,
        num_cpu_blocks=2048,
    )

    agent = Agent(
        backend=backend,
        name="local_jade",
        system_prompt="You are a helpful coding assistant.",
        tools=[calculator],
        verbose=True,
    )

    result = agent.run("Calculate the sum of first 100 prime numbers")
    print(f"\nAnswer: {result.answer}")


# ─── Example 4: Streaming ──────────────────────────────────────────────────

def example_streaming():
    """Stream tokens in real-time."""
    backend = OpenAICompatBackend(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key="YOUR_GROQ_KEY",
    )

    agent = Agent(
        backend=backend,
        name="stream_jade",
        system_prompt="You are a storyteller.",
        verbose=False,
    )

    print("📖 Streaming story:\n")
    for event in agent.stream_run("Tell me a short story about a brave AI"):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "answer":
            print("\n\n✅ Done!")


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    examples = {
        "chat": example_simple_chat,
        "tools": example_tool_agent,
        "local": example_megagemm_local,
        "stream": example_streaming,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("JadeAgent Examples")
        print("=" * 40)
        print("Usage: python examples/simple_chat.py <example>")
        print(f"Available: {', '.join(examples.keys())}")
        print()
        print("Running 'tools' example...")
        example_tool_agent()
