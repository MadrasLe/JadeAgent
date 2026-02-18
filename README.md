# 🐉 JadeAgent

**Next-Gen Agent Framework powered by MegaGemm**

Run autonomous AI agents with local GPU inference (MegaGemm) or any cloud API — zero vendor lock-in.

---

## ✨ Features

- 🔥 **MegaGemm Local Inference** — Zero latency, KV cache persistence, CPU offload
- 🔌 **Universal API Backend** — Works with OpenAI, Groq, Anthropic, Ollama, Together, any OpenAI-compatible endpoint
- 🔧 **@tool Decorator** — Auto JSON schema from type hints, safe mode
- 🔄 **ReAct Agent Loop** — Plan → Act → Reflect with tool calling
- 💬 **Multi-turn Sessions** — KV cache reuse across turns (MegaGemm)
- 📡 **Streaming** — Token-by-token streaming from any backend
- 🔑 **Key Rotation** — Automatic API key rotation on rate limits (from JadeHeavy)

### Coming Soon (Phase 2)
- 📊 **Graph Orchestration** — LangGraph-inspired state machine workflows
- 🧠 **Mixture of Agents** — MoA from ICLR 2025 paper
- 💬 **Multi-Agent Debate** — MAD from ACL 2024
- 🌳 **Tree of Thought** — ToT + Validator agent
- 💎 **ShoreStone v2** — Persistent vector memory with RFR curation

---

## 🚀 Quick Start

### Installation

```bash
# From source
cd JadeAgent
pip install -e .

# With MegaGemm local inference
pip install -e ".[local]"

# With memory system
pip install -e ".[all]"
```

### Simple Chat (Groq API)

```python
from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend

backend = OpenAICompatBackend(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_YOUR_KEY",
)

agent = Agent(backend=backend, name="jade")
print(agent.chat("What is quantum computing?"))
```

### Agent with Tools

```python
from jadeagent import Agent, tool
from jadeagent.backends import OpenAICompatBackend

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))

agent = Agent(
    backend=OpenAICompatBackend("gpt-4o", api_key="sk-..."),
    tools=[calculator],
)

result = agent.run("What is 15% of 340?")
print(result.answer)       # "51.0"
print(result.steps)        # 2 (think → calc → answer)
print(result.tool_calls_made)  # [ToolCall(calculator, {"expression": "340*0.15"})]
```

### Local GPU (MegaGemm)

```python
from jadeagent import Agent
from jadeagent.backends.megagemm import MegaGemmBackend

backend = MegaGemmBackend(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantize="int8",
    kv_offload=True,
    num_cpu_blocks=2048,
)

agent = Agent(backend=backend, tools=[calculator])
result = agent.run("Calculate sum of squares from 1 to 10")
```

### Any Provider (OpenAI-Compatible)

```python
from jadeagent.backends import OpenAICompatBackend

# Groq
groq = OpenAICompatBackend("llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# OpenAI
openai = OpenAICompatBackend("gpt-4o", api_key="sk-...")

# Ollama (local)
ollama = OpenAICompatBackend("qwen3:8b",
    base_url="http://localhost:11434/v1")

# Together AI
together = OpenAICompatBackend("meta-llama/Llama-3.1-70B",
    base_url="https://api.together.xyz/v1", api_key="...")

# Multiple keys with rotation (from JadeHeavy)
groq_rotated = OpenAICompatBackend("kimi-k2",
    base_url="https://api.groq.com/openai/v1",
    api_keys=["gsk_key1", "gsk_key2", "gsk_key3"])
```

---

## 🏗️ Architecture

```
JadeAgent Framework
├── backends/           # LLM providers
│   ├── megagemm.py     # ★ Local GPU (MegaGemm engine)
│   └── openai_compat.py# Any OpenAI-compatible API
├── core/               # Agent primitives
│   ├── agent.py        # ReAct loop
│   ├── session.py      # Multi-turn + KV persistence
│   ├── tools.py        # @tool decorator
│   └── streaming.py    # Token streaming
├── graph/              # Graph orchestration (Phase 2)
├── council/            # Multi-agent strategies (Phase 2)
└── memory/             # Persistent memory (Phase 2)
```

---

## 📚 Research References

| Paper | Year | Used In |
|-------|------|---------|
| MoA — Mixture-of-Agents | ICLR 2025 | `council/moa.py` |
| Multi-Agent Debate (MAD) | ACL 2024 | `council/debate.py` |
| Multi-Agent ToT Validator | 2024 | `council/tot.py` |
| ReAct: Reasoning + Acting | 2023 | `core/agent.py` |

---

## 📜 License

MIT
