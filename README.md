<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/795a82bd-11c1-49cc-a8ca-74d7fbdd760c" />

# Jade Agents

Este repositório contém diferentes agentes de IA desenvolvidos com propósitos específicos, utilizando a API da Groq para inferência rápida.

---

## 🤖 1. J.A.D.E. (jade/)

J.A.D.E. (Just Another Digital Entity) é uma agente de inteligência artificial multimodal projetada para interações naturais via texto, áudio e imagem. Ela utiliza a API da Groq para processamento rápido de linguagem e integra modelos especializados para visão computacional e síntese de voz.

### 🌟 Funcionalidades

- **Interação Multimodal**:
  - **Texto**: Chat conversacional inteligente.
  - **Áudio**: Transcrição de voz usando Whisper (via API Groq) e resposta via texto e áudio.
  - **Imagem**: Análise e descrição de imagens utilizando modelos BLIP, permitindo perguntas contextuais sobre o conteúdo visual.
- **Text-to-Speech (TTS)**: Respostas faladas utilizando `gTTS` (Google Text-to-Speech).
- **Memória de Longo Prazo (Simplificada)**: Mantém o contexto da conversa ativo, gerenciando o histórico para otimizar tokens.
- **Integração Google Colab**: Projetada com suporte nativo para upload de arquivos (`files.upload`) em ambientes notebook.

### 🛠️ Instalação e Uso (J.A.D.E.)

1.  Instale as dependências:
    ```bash
    pip install -r jade/requirements.txt
    ```
2.  Configure a chave da API:
    ```bash
    export GROQ_API_KEY="sua_chave_aqui"
    ```
3.  Execute:
    ```bash
    python jade/main.py
    ```

---

## 👨‍💻 2. CodeJade (code_jade/)

CodeJade é um assistente de programação avançado, projetado para atuar como um "pair programmer" inteligente, ideal para ambientes **Google Colab (ColabVM)** ou localmente.

### 🌟 Funcionalidades

- **Assistente de Código (ReAct)**: Utiliza um ciclo de raciocínio (Thought/Action/Observation) para resolver tarefas complexas.
- **Ferramentas Integradas**: Execução de shell, manipulação de arquivos e execução de Python.
- **Code Reviewer**: Um módulo de segurança que analisa o código gerado antes de salvar.
- **Integração Groq**: Utiliza modelos Llama 3 para inferência rápida.

### 🛠️ Instalação e Uso (CodeJade)

1.  Instale as dependências:
    ```bash
    pip install -r code_jade/requirements.txt
    ```
2.  Configure a chave da API (suporta `google.colab.userdata`):
    ```bash
    export GROQ_API_KEY="sua_chave_aqui"
    ```
3.  Execute:
    ```bash
    python code_jade/main.py
    ```

---

## 📚 3. JadeScholar (JadeScholar/)

**Agente Acadêmico e de Pesquisa**
- Focado em processamento de documentos e geração de material de estudo.
- Gera resumos, flashcards (Anki), podcasts e mapas mentais.
