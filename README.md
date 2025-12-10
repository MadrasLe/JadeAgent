<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/795a82bd-11c1-49cc-a8ca-74d7fbdd760c" />

# J.A.D.E. (Joint Agent Decision Engine)

J.A.D.E. is a modular framework for orchestrating autonomous AI agents. It implements specific architectures for multimodal interaction, automated code generation, and complex reasoning tasks using open-source models (via Groq).

---

##  Live Demo
**Interact with J.A.D.E. directly in your browser:**
### [👉 Launch Live Demo 👈](https://gabrielyukio2205-lgtm.github.io/github.io/)

---

## 🏗️ Jade Heavy: Cognitive Orchestration

**Jade Heavy** is an orchestration layer that decouples reasoning from direct model generation. Instead of relying on a single inference pass, it constructs a dynamic graph of thought processes.

### Architecture: ToT + MoA + CoT
The system implements a hybrid architecture:
1.  **Tree of Thoughts (ToT)**: The engine spawns multiple concurrent execution branches. Each branch explores a distinct solution path for the given problem.
2.  **Mixture of Agents (MoA)**: Specialized agent personas (e.g., "Critic", "Synthesizer") are injected into these branches to validate intermediate steps.
3.  **Chain of Thought (CoT)**: Strict prompting enforces step-by-step logic within each node of the tree.

### Preliminary Benchmarks (GPQA Diamond)
*Performance metrics using open-source models(kimi k2/gpt 120b oss/DeepSeek/Mistral):*

*   **Jade Heavy Low(Without Tool) (3-5 Branches)**:
    *   **Accuracy**: **~85-87%**
    *   **Significance**: Achieves results comparable to proprietary frontier models, significantly outperforming the 60-70% baseline of the underlying base models.
*   **Jade Heavy High(with Tools) (7 Branches)**:
    *   **Accuracy**: **94-95%** (on subset samples)
    *   **Mechanism**: The higher branch count increases the probability of finding a correct solution path in the search space, which the consensus mechanism then selects.

---

## 🧩 Technical Architecture by Module

### 1. 💻 CodeJade (`code_jade/`)
*Autonomous Software Engineering Agent*

CodeJade is designed to close the loop between code generation and execution.

*   **Execution Environment (`ToolManager`)**:
    *   **`execute_shell(command)`**: Runs bash commands via `subprocess` with a configurable timeout to prevent hanging processes.
    *   **`run_python(code)`**: Executes Python code in a restricted namespace.
        *   *Security*: The global dictionary is limited to `__builtins__`, `math`, `json`, and `os` (restricted). It prevents access to the agent's internal state instances.
    *   **`_confirm(action, content)`**: A safety gate that requires user approval for high-risk actions (`write_file`, shell execution) when `safe_mode` is enabled.

*   **The Guardian (`CodeReviewer`)**:
    *   Intercepts every `write_file` attempt.
    *   **Workflow**: The content is sent to a secondary LLM instance with a dedicated System Prompt focusing on security (SQLi, Arbitrary Execution) and logic bugs.
    *   **Protocol**: The reviewer must return a strict JSON payload `{ "status": "APPROVED" | "REJECTED", "feedback": "..." }`. Rejections block the file write and feed the error back to the coding agent for self-correction.

*   **Resiliency**:
    *   **Parsing**: Uses regex `r'\{.*\}'` with `re.DOTALL` to robustly extract JSON tool calls buried in verbose LLM text or Markdown blocks.
    *   **Memory**: Implements a sliding window context manager (`_manage_memory`) that preserves the System Prompt while truncating the oldest message pairs to fit within the context window (default: 20 turns).

### 2. J.A.D.E. Multimodal (`jade3/`)
*Interactive Voice & Vision Agent*

*   **Cognitive Loop (`JadeAgent.respond`)**:
    1.  **Remember**: Queries `ShoreStoneMemory` for semantically relevant past interactions.
    2.  **See**: If an image is present, `ImageHandler` generates a caption using BLIP.
    3.  **Respond**: Generates text response via Groq.
    4.  **Memorize**: Stores the new interaction embedding.
    5.  **Maintain**: Every `maintenance_interval` (10 turns), triggers the Heuristic Curator.

*   **ShoreStone Memory System (`shorestone.py`)**:
    *   **Vector Store**: Uses `chromadb.PersistentClient` for local storage.
    *   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional vectors.
    *   **Compression**: Supports optional PCA (Principal Component Analysis) via `joblib` to reduce vector dimensionality if a model is trained.

*   **Heuristic Curator (`curator_heuristic.py`)**:
    *   Implements the **RFR-Score** algorithm to decide which memories to keep:
        $$ Score = \alpha \cdot \log(1+Frequency) + \beta \cdot e^{-\lambda \cdot Recency} + \gamma \cdot Relevance $$
    *   **Geometric Relevance**: Calculates cosine similarity to find the "neighborhood" of a memory. Isolated memories (low similarity to others) are penalized.

### 3. 🎓 Jade Scholar (`JadeScholar/`)
*Graph-Based Educational Agent*

*   **Graph State (`GraphState`)**:
    *   A shared data object that flows between agents, holding `raw_content`, `summary`, `script`, and `quiz_data`.
    
*   **Pipeline**:
    1.  **Ingestion**: `IngestAgent` detects `http` vs `.pdf` paths.
    2.  **Synthesis**: `ProfessorAgent` transforms raw text into didactic summaries using structured Markdown prompts.
    3.  **Scripting**: `ScriptwriterAgent` generates a JSON list of dialogue objects `[{"speaker": "Gabriel", "text": "..."}]`.
    4.  **Audio Synthesis**: The `ToolBox.generate_audio_mix` method iterates through the script. It uses `gTTS` for base audio and `pydub` to merge segments, applying silence intervals for pacing.
    5.  **Evaluation**: `ExaminerAgent` creates a quiz JSON and runs a terminal-based interactive loop.

---

##  Installation & Dependencies

### Prerequisites
*   **Python 3.8+**
*   **FFmpeg**: Required for audio processing (`pydub`/`gTTS`).
    *   `sudo apt-get install ffmpeg` (Debian/Ubuntu)

### Setup Steps

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/jade-agents.git
    cd jade-agents
    ```

2.  **Install Python Packages**:
    ```bash
    pip install groq pypdf gtts pydub beautifulsoup4 requests fpdf chromadb sentence-transformers joblib scikit-learn transformers torch pillow
    ```

3.  **Configure Credentials**:
    Set your Groq API key in your environment variables:
    ```bash
    export GROQ_API_KEY="gsk_..."
    ```

---

## ⚙️ Configuration Reference

### `jade3/config.json`
| Key | Default | Description |
| :--- | :--- | :--- |
| `groq_model` | `moonshotai/kimi-k2-instruct-0905` | The LLM inference engine. |
| `language` | `pt/en` | TTS and System Prompt language. |
| `maintenance_interval` | `10` | Interaction turns before running memory curation. |
| `caption_model` | `Salesforce/blip...` | HuggingFace model ID for vision. |

### `code_jade/config.json`
| Key | Default | Description |
| :--- | :--- | :--- |
| `safe_mode` | `true` | If true, asks for Y/N confirmation before shell/file ops. |
| `work_dir` | `./workspace` | Directory sandbox for file creation. |
| `max_context` | `20` | Sliding window size (number of messages). |

---

## 📜 License

