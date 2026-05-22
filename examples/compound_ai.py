"""
JadeAgent — Example: Compound AI Patterns (Phase 2)

Demonstrates:
1. Graph-based workflow with cycles
2. MoA (Mixture of Agents) — multiple proposers + aggregator
3. Multi-Agent Debate — debaters + judge
4. Pipeline — sequential refinement
5. Buffer Memory
"""

from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend
from jadeagent.graph import StateGraph, START, END
from jadeagent.council import MixtureOfAgents, Debate, Pipeline, TreeOfThought
from jadeagent.memory import BufferMemory


# ─── Setup — create Groq backends with different models ────────────────────

def make_backend(model="llama-3.1-8b-instant", api_key="YOUR_GROQ_KEY"):
    return OpenAICompatBackend(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


# ─── Example 1: Graph Workflow ─────────────────────────────────────────────

def example_graph():
    """Research workflow with conditional looping."""
    from typing import TypedDict

    class ResearchState(TypedDict):
        query: str
        searches: list
        analysis: str
        iteration: int

    backend = make_backend()

    def search_node(state):
        agent = Agent(backend, name="searcher", verbose=False)
        result = agent.chat(
            f"Research this topic and provide key facts: {state['query']}\n"
            f"Iteration: {state.get('iteration', 0) + 1}"
        )
        return {
            "searches": [result],
            "iteration": state.get("iteration", 0) + 1,
        }

    def analyze_node(state):
        agent = Agent(backend, name="analyst", verbose=False)
        all_research = "\n\n".join(state["searches"])
        result = agent.chat(
            f"Analyze this research and provide a comprehensive summary:\n\n{all_research}"
        )
        return {"analysis": result}

    def should_search_more(state):
        if state.get("iteration", 0) < 2:
            return "search"  # Loop back for more research
        return "analyze"     # Enough research, move to analysis

    graph = StateGraph()
    graph.add_node("search", search_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "search")
    graph.add_conditional_edge("search", should_search_more)
    graph.add_edge("analyze", END)

    result = graph.compile().run(
        {"query": "Impact of AI on healthcare", "searches": [], "iteration": 0},
        verbose=True,
    )
    print(f"\n📊 Analysis:\n{result['analysis'][:500]}")


# ─── Example 2: MoA ────────────────────────────────────────────────────────

def example_moa():
    """Mixture of Agents: 3 proposers → aggregator."""
    api_key = "YOUR_GROQ_KEY"

    moa = MixtureOfAgents(
        proposers=[
            Agent(make_backend("llama-3.1-8b-instant", api_key),
                  name="llama", verbose=False),
            Agent(make_backend("gemma2-9b-it", api_key),
                  name="gemma", verbose=False),
            Agent(make_backend("llama-3.1-8b-instant", api_key),
                  name="mixtral", verbose=False),
        ],
        aggregator=Agent(
            make_backend("llama-3.1-8b-instant", api_key),
            name="aggregator",
            system_prompt="Synthesize the best answer from multiple proposals. Be comprehensive.",
            verbose=False,
        ),
        num_layers=2,
    )

    answer = moa.run("Explain the significance of attention mechanisms in deep learning")
    print(f"\n🧠 MoA Answer:\n{answer[:500]}")


# ─── Example 3: Debate ─────────────────────────────────────────────────────

def example_debate():
    """Multi-Agent Debate: 2 debaters + judge."""
    api_key = "YOUR_GROQ_KEY"

    debate = Debate(
        debaters=[
            Agent(make_backend(api_key=api_key), name="proponent",
                  system_prompt="You argue strongly IN FAVOR of the given position. Support with evidence.",
                  verbose=False),
            Agent(make_backend(api_key=api_key), name="opponent",
                  system_prompt="You argue AGAINST the given position. Be critical and analytical.",
                  verbose=False),
        ],
        judge=Agent(
            make_backend(api_key=api_key), name="judge",
            system_prompt="You are a fair and analytical judge. Evaluate arguments objectively.",
            verbose=False,
        ),
        rounds=2,
    )

    verdict = debate.run("Should AI development be regulated by governments?")
    print(f"\n⚖️ Verdict:\n{verdict[:500]}")


# ─── Example 4: Pipeline ───────────────────────────────────────────────────

def example_pipeline():
    """Sequential pipeline: draft → review → polish."""
    api_key = "YOUR_GROQ_KEY"
    backend = make_backend(api_key=api_key)

    pipeline = Pipeline([
        Agent(backend, name="drafter",
              system_prompt="Write a first draft. Be creative and comprehensive.",
              verbose=False),
        Agent(backend, name="reviewer",
              system_prompt="Review this text. Fix errors, improve clarity, add missing points.",
              verbose=False),
        Agent(backend, name="polisher",
              system_prompt="Polish this text. Make it engaging, well-structured, and publication-ready.",
              verbose=False),
    ])

    result = pipeline.run("Write a short blog post about why Compound AI systems beat monolithic models")
    print(f"\n📝 Final:\n{result[:500]}")


# ─── Example 5: Memory ─────────────────────────────────────────────────────

def example_memory():
    """Agent with buffer memory."""
    backend = make_backend()
    memory = BufferMemory(max_size=50)

    # Pre-load some memories
    memory.memorize("Gabriel is a CS student studying AI/ML")
    memory.memorize("He built MegaGemm, a custom CUDA inference engine")
    memory.memorize("His project uses PyTorch and Triton for GPU kernels")
    memory.memorize("He prefers Python and is 23 years old")

    # Retrieve relevant memories
    results = memory.remember("What programming tools does Gabriel use?", k=3)
    print("🧠 Relevant memories:")
    for r in results:
        print(f"  → {r}")


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    examples = {
        "graph": example_graph,
        "moa": example_moa,
        "debate": example_debate,
        "pipeline": example_pipeline,
        "memory": example_memory,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("JadeAgent Phase 2 Examples")
        print("=" * 40)
        print(f"Usage: python examples/compound_ai.py <example>")
        print(f"Available: {', '.join(examples.keys())}")
        print()
        print("Running 'memory' example (no API needed)...")
        example_memory()
