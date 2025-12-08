<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/795a82bd-11c1-49cc-a8ca-74d7fbdd760c" />

# Jade Agents

## LIVE DEMO: https://gabrielyukio2205-lgtm.github.io/github.io/

This repository hosts a collection of specialized AI agents developed for distinct purposes, leveraging the Groq API for ultra-fast inference and high-performance capabilities.

---

##  **Jade Heavy (Preliminary)**

**Jade Heavy** is an advanced agent orchestration architecture combining **Tree of Thoughts (ToT)**, **Mixture of Agents (MoA)**, and **Chain of Thought (CoT)** methodologies to solve complex, high-reasoning problems.

###  Key Architecture
- **Orchestration**: Integrates multiple reasoning paths (ToT) with collaborative agent synthesis (MoA) and step-by-step logic (CoT).
- **Goal**: Tackle difficult benchmarks and real-world problems that typically stump single-model approaches.

### 📊Preliminary Results
Jade Heavy has demonstrated state-of-the-art performance even when utilizing open-source models:

- **Jade Heavy Low (3 Branches)**: 
  - Achieved approximately **87% accuracy** on **GPQA Diamond**.
  - *Context*: The open-source models typically perform between **60-70%** on this benchmark in the tests.
  
- **Jade Heavy High (7 Branches)**: 
  - Currently reaching **94-95% accuracy** on subset sample tests.

---

## 1. J.A.D.E. (jade/)

**J.A.D.E. (Joint Agent Decision Engine)** is a multimodal AI agent designed for natural interaction via text, audio, and images. It orchestrates specialized models for computer vision and voice synthesis, powered by Groq's LLMs, Openrouter and Mistral.

###  Features

- **Multimodal Interaction**:
  - **Text**: Intelligent conversational chat.
  - **Audio**: Voice transcription using Whisper (via Groq API) and spoken responses.
  - **Image**: Visual analysis and description using BLIP models, allowing for contextual questions about images.
- **Text-to-Speech (TTS)**: Natural spoken responses using `gTTS` (Google Text-to-Speech).
- **Memory Management**: Sliding window context management to optimize token usage while maintaining conversation history.
- **Colab Integration**: Native support for file uploads and audio processing within Google Colab environments.

### 🛠️ Installation & Usage

1.  Install dependencies:
    ```bash
    pip install -r jade/requirements.txt
    ```
2.  Set up your API key:
    ```bash
    export GROQ_API_KEY="your_key_here"
    ```
3.  Run the agent:
    ```bash
    python jade/main.py
    ```

---

## 2. CodeJade (code_jade/)

**CodeJade** is an advanced coding assistant acting as an intelligent "pair programmer." It uses a ReAct (Reasoning + Acting) loop to autonomously solve programming tasks, debug code, and manage files.

### Features

- **ReAct Agent**: cycles through **Thought**, **Action**, and **Observation** to solve complex engineering tasks.
- **Integrated Toolbelt**:
  - **Shell Execution**: Run bash commands safely.
  - **File Manipulation**: Read, write, and manage codebase files.
  - **Python Execution**: Run Python scripts in a sandboxed environment.
- **Code Reviewer**: A built-in security module that reviews generated code for quality and safety before applying changes to the file system.
- **Environment**: Optimized for **Google Colab (ColabVM)** and VM development.

### 🛠️ Installation & Usage

1.  Install dependencies:
    ```bash
    pip install -r code_jade/requirements.txt
    ```
2.  Set up your API key:
    ```bash
    export GROQ_API_KEY="your_key_here"
    ```
3.  Run the assistant:
    ```bash
    python code_jade/main.py
    ```

---

## 3. JadeScholar (JadeScholar/)

**JadeScholar** is a dedicated academic and research agent designed to transform raw information into structured educational materials.

###  Features

- **Knowledge Graph Generation**: Creates visual mind maps connecting complex concepts.
- **Multi-Source Ingestion**: Processes PDFs, text files, YouTube videos, and web search results.
- **Educational Output**:
  - **Summaries & Debates**: Generates comprehensive summaries and simulated debates between AI personas.
  - **Flashcards**: Automatically creates Anki decks (`.apkg`) for spaced repetition learning.
  - **Podcasts**: Synthesizes audio discussions for on-the-go learning.
  - **Quizzes**: Generates assessment materials to test understanding.
