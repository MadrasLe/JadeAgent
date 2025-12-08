<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/795a82bd-11c1-49cc-a8ca-74d7fbdd760c" />

# J.A.D.E. (Joint Agent Decision Engine)

J.A.D.E. is a comprehensive ecosystem of specialized AI agents designed to handle diverse tasks ranging from complex reasoning and coding to autonomous education and multimodal interaction. This repository houses the various components of the J.A.D.E. project, demonstrating cutting-edge agent orchestration architectures.

## Live Demo
Check out the live demo here: **[J.A.D.E. Live Demo](https://gabrielyukio2205-lgtm.github.io/github.io/)**

---

## 🏗️ Jade Heavy: Advanced Reasoning Orchestrator

**Jade Heavy** represents the pinnacle of our agent orchestration research. It utilizes a sophisticated combination of **Tree of Thoughts (ToT)**, **Mixture of Agents (MoA)**, and **Chain of Thought (CoT)** architectures to solve highly difficult problems that stump traditional single-model approaches.

By spawning multiple reasoning branches and having agents critique and refine each other's outputs, Jade Heavy achieves state-of-the-art performance even with smaller underlying models.

### Preliminary Results

Jade Heavy has shown remarkable results in early benchmarks, significantly outperforming base models:

*   **Jade Heavy Low (3 branches)**:
    *   Achieved **~87% accuracy** on **GPQA Diamond**.
    *   *Note*: This result was achieved using weaker open-source models that typically perform in the 60-70% range on this benchmark.
*   **Jade Heavy High (7 branches)**:
    *   Achieved **94-95% accuracy** on subset samples.

This demonstrates the power of the **Joint Agent Decision Engine** architecture in amplifying the reasoning capabilities of open-source models through effective orchestration.

---

##  Agent Ecosystem

The J.A.D.E. repository contains several specialized agents, each capable of operating autonomously or as part of the larger system.

### J.A.D.E. (Multimodal)
*Located in: `jade3/`*

The flagship multimodal interaction agent, designed with a J.A.D.E persona. It integrates vision and audio models to create a seamless natural interface for original non-multimodal models.

*   **Core Capabilities**:
    *   **Vision-First Architecture**: Uses **BLIP (Salesforce/blip-image-captioning-large)/Florence 2** to "see" and describe images in real-time. It can analyze visual context and integrate it into its reasoning stream.
    *   **Natural Voice Interface**:
        *   **Hearing**: Processes audio inputs (integration ready via Groq/WhisperV3).
        *   **Speaking**: Features a simple TTS engine using **gTTS** with auto-play capabilities in Jupyter/Colab environments, allowing J.A.D.E. to vocalize responses instantly.
        *   **ShoreStone Memory System**: A persistent memory architecture with a **Heuristic Curator**.
        *   **Self-Maintenance**: The agent autonomously runs a "sleep cycle" (maintenance interval) every 10 interactions.
        *   **RFR-Score**: Uses a formula (Recency, Frequency, Relevance) to score memories, deciding what to keep, what to forget, and what to archive, mimicking human long-term memory consolidation.
        *   **Geometric Relevance**: Uses cosine similarity to find "neighborhoods" of related memories, ensuring contextually rich responses.

###  Jade Scholar
*Located in: `JadeScholar/`*

An autonomous educational agent designed to act as a personal AI tutor. It processes learning materials from various sources and converts them into interactive study formats.
*   **Features**:
    *   Ingests **PDFs**, **URLs**, and **Text**.
    *   Generates **didactic summaries** using a "Professor" persona.
    *   Creates and narrates **Podcasts** featuring distinct voices (e.g., Gabriel and Berta) using multi-speaker TTS and **Pydub** for audio mixing.
    *   Generates and conducts **interactive quizzes** to test your knowledge.
*   **Stack**: Groq (Llama 3), gTTS, pypdf, BeautifulSoup.


### CodeJade: The Autonomous Software Engineer
*Located in: `code_jade/`*

CodeJade is a coding assistant; it is a autonomous software engineering agent built to operate within local environments or Google Colab to help. It bridges the gap between LLM code generation and safe, practical code execution.

*   **Architecture**:
    *   **ToolManager**: A dedicated orchestration layer that handles filesystem operations (`write_file`, `read_file`, `list_files`), shell command execution, and python script execution. It abstracts the OS complexity from the language model.
    *   **Context-Aware**: Implements a sliding window memory system that manages token usage effectively, allowing for prolonged coding sessions without losing the thread of the conversation.(future update with ShoreStoneSWE)
    *   **Parsing**: Uses advanced regex-based parsing to reliably extract JSON tool calls from LLM responses, ensuring stability even when models hallucinate formatting inconsistencies.

*   **Safety & Security**:
    *   **Guardian CodeReviewer**: A secondary, specialized "Engineer" agent (`reviewer.py`) intercepts every file modification request. It performs a rigorous static analysis to check for logic bugs, security vulnerabilities (like SQL injection or unsafe exec usage), and code quality before any change is committed to the disk.
    *   **Sandboxed Execution**: Python code execution is confined to a restricted global scope, preventing accidental damage to the host system while still allowing necessary imports like `os`, `json`, and `math`.
    *   **Human-in-the-Loop**: The system includes a confirmation step (`_confirm`) for critical actions (Shell, File Write, Code Exec), giving the user final authority over what the agent does.
