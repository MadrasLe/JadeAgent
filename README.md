# 💎 Jade Project: A Revolução dos Agentes IA

Fala, dev! Bem-vindo ao **Jade Project**. Esse repositório é uma coleção de agentes autônomos de IA brabíssimos, cada um especialista em uma área. Se você quer estudar, codar ou ter uma assistente pessoal multimodal, você tá no lugar certo.

## 🚀 O que tem aqui?

O projeto é dividido em três núcleos de inteligência:

### 1. 🎓 JadeScholar (O Professor IA)
*Local:* `JadeScholar/scholar_graph.py`

Transforme qualquer conteúdo em uma aula completa. O **Scholar Graph Agent** é um sistema baseado em grafos projetado para rodar no Google Colab.

- **📥 Ingestão Universal:** Lê PDFs, sites (URLs) ou texto puro.
- **🧠 Professor Agent:** Gera resumos didáticos e explicativos.
- **🎙️ Podcast Generator:** Cria um podcast estilo "mesa redonda" com duas vozes distintas (Gabriel 🇧🇷 e Professora Berta 🇵🇹) debatendo o assunto.
- **📝 Examiner Agent:** Gera quizzes interativos para testar seu conhecimento.

**Como usar:**
Abra o `scholar_graph.py` no Google Colab, defina sua `GROQ_API_KEY` e execute. O script instala tudo sozinho.

---

### 2. 💻 CodeJade (Seu Pair Programmer)
*Local:* `code_jade/`

Um assistente de programação estilo **Cursor AI**, mas que roda no seu terminal. Ele não só escreve código, mas revisa e executa.

- **🛠️ Tool Manager:** Executa comandos shell, manipula arquivos e roda scripts Python.
- **🛡️ Code Reviewer:** Um módulo de segurança que intercepta e revisa qualquer código antes de salvar. Se o código for ruim ou perigoso, ele bloqueia!
- **⚡ ReAct Loop:** Raciocínio iterativo para resolver problemas complexos.

**Como rodar:**
```bash
# Configure sua chave
export GROQ_API_KEY="sua-chave-aqui"

# Instale as dependências
pip install -r code_jade/requirements.txt

# Execute
python code_jade/main.py
```

---

### 3. 🤖 J.A.D.E. (Assistente Multimodal)
*Local:* `jade/`

J.A.D.E. (Just Another Digital Entity? Talvez...) é uma assistente pessoal completa.

- **👁️ Visão:** Analisa e descreve imagens.
- **🗣️ Audição:** Entende comandos de voz.
- **💬 Fala:** Responde com Text-to-Speech (TTS) fluido.
- **🧠 Cérebro:** Powered by Groq/Llama 3.

**Como rodar:**
```bash
pip install -r jade/requirements.txt
python jade/main.py
```

## 🛠️ Configuração Geral

1. Clone o repositório.
2. Garanta que você tem Python 3.9+ instalado.
3. Obtenha uma chave de API na [Groq](https://groq.com).
4. Defina a variável de ambiente:
   ```bash
   export GROQ_API_KEY="gsk_..."
   ```

## 🤝 Contribuição

Curtiu? Manda aquele PR ou abre uma Issue. O código é livre!

---
*Feito com ⚡ e ☕.*
