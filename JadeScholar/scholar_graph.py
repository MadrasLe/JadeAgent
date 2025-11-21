# @title 🎓 Scholar Graph Agent - O Professor IA (Versão Colab)
# Este arquivo implementa um sistema de Agentes em Grafo para ensino autônomo.

import os
import sys
import json
import time
import re
import textwrap
from io import BytesIO
from typing import List, Dict, Any, Optional

# --- 1. Setup e Dependências (Auto-Instalação para Colab) ---
def setup_environment():
    packages = ["groq", "pypdf", "gtts", "pydub", "beautifulsoup4", "requests", "fpdf"]
    missing = []
    
    # Verifica pacotes Python
    for pkg in packages:
        try:
            __import__(pkg if pkg != "beautifulsoup4" else "bs4")
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 Instalando dependências ausentes: {', '.join(missing)}...")
        os.system(f"pip install -q {' '.join(missing)}")
    
    # Verifica FFmpeg (Necessário para Pydub/Audio)
    if not os.path.exists("/usr/bin/ffmpeg"):
        print("🎥 Instalando FFmpeg (Sistema)...")
        os.system('apt-get install -q ffmpeg')
    
    print("✅ Ambiente configurado com sucesso!")

# Executa setup se rodar direto
try:
    import groq
    import pypdf
    from gtts import gTTS
    from pydub import AudioSegment
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    setup_environment()
    import groq
    import pypdf
    from gtts import gTTS
    from pydub import AudioSegment
    import requests
    from bs4 import BeautifulSoup

from IPython.display import Audio, display, clear_output

# --- 2. Configuração Global ---
# Tenta pegar do ambiente ou define um placeholder
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "INSIRA_SUA_CHAVE_AQUI")

# --- 3. Camada de Ferramentas (Tooling Layer) ---

class ToolBox:
    """Caixa de ferramentas para os agentes."""
    
    @staticmethod
    def read_pdf(filepath: str) -> str:
        """Extrai texto de um arquivo PDF."""
        try:
            print(f"📄 [Ferramenta] Lendo PDF: {filepath}...")
            reader = pypdf.PdfReader(filepath)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return clean_text
        except Exception as e:
            return f"Erro ao ler PDF: {str(e)}"

    @staticmethod
    def scrape_web(url: str) -> str:
        """Extrai texto principal de uma URL."""
        try:
            print(f"🌐 [Ferramenta] Acessando URL: {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts e estilos
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            
            text = soup.get_text()
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return clean_text[:20000] # Limite de segurança
        except Exception as e:
            return f"Erro ao acessar web: {str(e)}"

    @staticmethod
    def generate_audio_mix(script: List[Dict], filename="aula_podcast.mp3"):
        """Gera áudio com vozes diferentes (Gabriel/BR e Berta/PT)."""
        print("🎙️ [Ferramenta] Produzindo áudio no estúdio...")
        combined = AudioSegment.silent(duration=500)
        
        for line in script:
            speaker = line.get("speaker", "Narrador").upper()
            text = line.get("text", "")
            
            # Lógica de Vozes
            if "BERTA" in speaker or "PROFESSORA" in speaker:
                # Voz PT-PT
                tts = gTTS(text=text, lang='pt', tld='pt', slow=False)
            else:
                # Voz PT-BR (Padrão)
                tts = gTTS(text=text, lang='pt', tld='com.br', slow=False)
            
            # Buffer em memória
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            try:
                segment = AudioSegment.from_file(fp, format="mp3")
                combined += segment
                combined += AudioSegment.silent(duration=400) # Pausa entre falas
            except Exception as e:
                print(f"Erro no segmento de áudio: {e}")

        combined.export(filename, format="mp3")
        return filename

# --- 4. Estado e Motor de IA ---

class GraphState:
    def __init__(self):
        self.raw_content: str = ""
        self.summary: str = ""
        self.script: List[Dict] = []
        self.quiz_data: List[Dict] = []
        self.history: List[Dict] = []

class LLMEngine:
    def __init__(self):
        # Tenta pegar chave do os.environ se não estiver na global
        api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
        self.client = groq.Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def chat(self, messages: List[Dict], json_mode=False) -> str:
        try:
            kwargs = {
                "messages": messages,
                "model": self.model,
                "temperature": 0.6
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na IA: {e}"

# --- 5. Agentes (Nós do Grafo) ---

class IngestAgent:
    """Especialista em Entrada: Identifica e extrai conteúdo."""
    def process(self, user_input: str) -> str:
        if user_input.lower().endswith(".pdf") and os.path.exists(user_input):
            return ToolBox.read_pdf(user_input)
        elif user_input.startswith("http"):
            return ToolBox.scrape_web(user_input)
        else:
            return user_input # Texto puro

class ProfessorAgent:
    """Especialista em Ensino: Resume e Explica."""
    def __init__(self, llm: LLMEngine):
        self.llm = llm
    
    def summarize(self, text: str) -> str:
        print("🧠 [Professor] Analisando e resumindo conteúdo...")
        prompt = f"""
        Você é um Professor Especialista. Resuma o seguinte texto de forma didática, 
        destacando os pontos chave para um aluno. Use Markdown com tópicos e negrito.
        
        Texto: {text[:25000]}
        """
        return self.llm.chat([{"role": "user", "content": prompt}])

class ScriptwriterAgent:
    """Especialista em Roteiro: Cria diálogos."""
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_script(self, content: str) -> List[Dict]:
        print("✍️ [Roteirista] Criando roteiro de Podcast...")
        prompt = f"""
        Crie um roteiro de podcast educativo (aprox 8 falas).
        Personagens:
        - GABRIEL (Aluno Brasileiro, curioso, faz perguntas pertinentes).
        - BERTA (Professora Portuguesa, sábia, explica com clareza).
        
        Baseado nisto: {content[:20000]}
        
        SAÍDA OBRIGATÓRIA: JSON formato lista de objetos:
        {{ "dialogue": [ {{"speaker": "Gabriel", "text": "..."}}, {{"speaker": "Berta", "text": "..."}} ] }}
        """
        response = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        try:
            data = json.loads(response)
            return data.get("dialogue", [])
        except:
            print("Erro no JSON do roteiro, tentando fallback...")
            return [{"speaker": "Berta", "text": "Desculpe, houve um erro ao gerar o roteiro."}]

class ExaminerAgent:
    """Especialista em Avaliação: Cria e Aplica Provas."""
    def __init__(self, llm: LLMEngine):
        self.llm = llm
        
    def generate_quiz(self, content: str, num_questions=5) -> List[Dict]:
        print("📝 [Examinador] Elaborando questões de prova...")
        prompt = f"""
        Crie um Quiz com {num_questions} perguntas de múltipla escolha sobre o texto.
        Nível: Desafiador mas justo.
        
        SAÍDA JSON OBRIGATÓRIA:
        {{
            "quiz": [
                {{
                    "question": "Pergunta?",
                    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                    "correct_option": "A",
                    "explanation": "Explicação curta."
                }}
            ]
        }}
        
        Texto: {content[:20000]}
        """
        response = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        try:
            data = json.loads(response)
            return data.get("quiz", [])
        except:
            return []

    def interactive_session(self, quiz_data: List[Dict]):
        """Roda o loop interativo da prova."""
        score = 0
        print("\n" + "="*40)
        print("🎓 INICIANDO PROVA INTERATIVA")
        print("="*40)
        
        for i, q in enumerate(quiz_data):
            print(f"\nQUESTÃO {i+1}: {q['question']}")
            for opt in q['options']:
                print(opt)
            
            ans = input("\nSua resposta (A/B/C/D): ").strip().upper()
            
            # Verifica só a primeira letra
            correct_letter = q['correct_option'][0].upper()
            user_letter = ans[0].upper() if ans else "X"
            
            if user_letter == correct_letter:
                print(f"✅ CORRETO! {q['explanation']}")
                score += 1
            else:
                print(f"❌ ERRADO. A correta era {q['correct_option']}. \nExplicação: {q['explanation']}")
            
            time.sleep(1)
        
        print(f"\n🏆 RESULTADO FINAL: {score}/{len(quiz_data)}")
        if score == len(quiz_data):
            print("🌟 PERFEITO! Você dominou o assunto.")
        elif score >= len(quiz_data)/2:
            print("👍 BOM TRABALHO. Continue estudando.")
        else:
            print("📚 PRECISAS ESTUDAR MAIS.")

# --- 6. Orquestrador (O Gerente do Grafo) ---

class ScholarGraph:
    def __init__(self):
        self.state = GraphState()
        # Instancia LLM. Se der erro de chave, o loop principal trata.
        try:
            self.llm = LLMEngine()
        except:
            self.llm = None
        
        # Inicializa Agentes
        self.ingestor = IngestAgent()
        if self.llm:
            self.professor = ProfessorAgent(self.llm)
            self.scriptwriter = ScriptwriterAgent(self.llm)
            self.examiner = ExaminerAgent(self.llm)
    
    def run(self):
        print("\n🎓 SCHOLAR GRAPH - Seu Assistente de Aprendizado IA")
        print("---------------------------------------------------")
        
        # Checagem de API KEY Dinâmica
        if "GROQ_API_KEY" not in os.environ and "GROQ_KEY" not in os.environ and GROQ_API_KEY == "INSIRA_SUA_CHAVE_AQUI":
            print("\n🔑 CHAVE API NECESSÁRIA")
            key = input("Cole sua GROQ_API_KEY: ").strip()
            if key:
                os.environ["GROQ_API_KEY"] = key
                self.llm = LLMEngine()
                self.professor = ProfessorAgent(self.llm)
                self.scriptwriter = ScriptwriterAgent(self.llm)
                self.examiner = ExaminerAgent(self.llm)
            else:
                print("❌ Sem chave, não posso funcionar. Encerrando.")
                return

        # 1. Input
        target = input("\n📂 Digite o Caminho do PDF, URL ou cole o Texto: ").strip()
        if not target:
             print("Entrada vazia. Encerrando.")
             return

        self.state.raw_content = self.ingestor.process(target)
        
        if not self.state.raw_content or len(self.state.raw_content) < 10:
            print("❌ Erro: Conteúdo muito curto ou inválido/vazio.")
            return

        print("\n✅ Conteúdo processado com sucesso!")
        
        while True:
            print("\n" + "-"*30)
            print(" MENU PRINCIPAL")
            print("-"*30)
            print("1. 🧠 Gerar Resumo Didático")
            print("2. 🎙️ Criar Podcast (Áudio)")
            print("3. 📝 Fazer Prova (Quiz Interativo)")
            print("4. 🚪 Sair")
            
            choice = input("\nEscolha uma opção: ").strip()
            
            if choice == "1":
                summary = self.professor.summarize(self.state.raw_content)
                self.state.summary = summary
                print("\n📄 RESUMO DO PROFESSOR:\n")
                print(summary)
                input("\nPressione Enter para voltar ao menu...")
                
            elif choice == "2":
                script = self.scriptwriter.create_script(self.state.raw_content)
                self.state.script = script
                filename = ToolBox.generate_audio_mix(script)
                print(f"\n🎧 Podcast gerado: {filename}")
                
                # Tenta tocar no Colab/Jupyter
                try:
                    display(Audio(filename, autoplay=True))
                except:
                    print(f"Arquivo salvo em: {filename}")
                
            elif choice == "3":
                quiz = self.examiner.generate_quiz(self.state.raw_content)
                self.state.quiz_data = quiz
                if quiz:
                    self.examiner.interactive_session(quiz)
                else:
                    print("Erro ao gerar quiz.")
                input("\nPressione Enter para voltar ao menu...")
                
            elif choice == "4":
                print("Até logo! Bons estudos. 🍎")
                break
            else:
                print("Opção inválida.")

if __name__ == "__main__":
    app = ScholarGraph()
    app.run()
