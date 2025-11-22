import json
import logging
from groq import Groq
from .tools import ToolManager
from .reviewer import CodeReviewer

class CodeJadeAgent:
    def __init__(self, config_path="code_jade/config.json"):
        # Carrega Configuração
        try:
            with open(config_path) as f:
                self.cfg = json.load(f)
        except FileNotFoundError:
            # Fallback se não achar, mas idealmente deve existir
            self.cfg = {"groq_model": "llama3-70b-8192", "safe_mode": True, "work_dir": "./workspace", "max_context": 20}

        self.client = Groq(api_key=self._get_api_key())
        self.tools = ToolManager(safe_mode=self.cfg.get("safe_mode", True), work_dir=self.cfg.get("work_dir", "."))
        self.reviewer = CodeReviewer(self.cfg)
        
        # System Prompt focado em Code Assistant
        self.system_prompt = """Você é CodeJade, um assistente de programação avançado (estilo Cursor AI).
Seu objetivo é ajudar o usuário a escrever código, corrigir bugs e explorar o projeto.

FERRAMENTAS DISPONÍVEIS:
Você tem acesso a ferramentas. Para usá-las, você DEVE responder APENAS com um bloco JSON estrito no seguinte formato:
{"tool": "nome_da_ferramenta", "args": {"arg1": "valor1"}}

As ferramentas são:
1. execute_shell(command: str) -> Executa comandos bash (ls, pip, git, etc).
2. read_file(filepath: str) -> Lê o conteúdo de um arquivo.
3. write_file(filepath: str, content: str) -> Cria ou sobrescreve um arquivo.
4. list_files(path: str) -> Lista arquivos.
5. run_python(code: str) -> Executa script Python.

REGRAS:
- Se precisar de informações, use 'list_files' ou 'read_file'.
- Se precisar rodar algo, use 'execute_shell' ou 'run_python'.
- Se for apenas conversar ou explicar, responda em texto normal (sem JSON).
- Mantenha respostas diretas e técnicas.
- Sempre verifique se o código funciona rodando-o se possível.

EXEMPLO DE USO:
Usuário: "Crie um hello world em python"
CodeJade: {"tool": "write_file", "args": {"filepath": "hello.py", "content": "print('Hello World')"}}
(Sistema executa e retorna sucesso)
CodeJade: "Arquivo criado. Quer que eu execute?"
"""
        self.history = [{"role": "system", "content": self.system_prompt}]
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _get_api_key(self):
        import os
        key = os.getenv("GROQ_API_KEY")
        if not key:
            # Tenta pegar do colab se estiver lá
            try:
                from google.colab import userdata
                key = userdata.get('GROQ_API_KEY')
            except:
                pass
        if not key:
            print("⚠️ AVISO: GROQ_API_KEY não encontrada. O agente pode falhar.")
            return "dummy_key"
        return key

    def _chat(self, messages):
        """Chama o modelo Groq."""
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.cfg["groq_model"],
                temperature=0.3, # Baixa temperatura para código preciso
                stop=None
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"❌ Erro na API Groq: {e}"

    def process_tool_call(self, response_text):
        """Tenta parsear JSON para chamada de ferramenta usando Regex."""
        import json
        import re
        
        try:
            # Procura pelo padrão JSON: { ... }
            # O regex considera chaves aninhadas simples, mas foca no bloco principal
            # A opção re.DOTALL permite que o ponto (.) case com novas linhas
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                return None
            
            json_str = match.group(0)
            
            # Tenta carregar o JSON
            data = json.loads(json_str)
            
            if "tool" in data and "args" in data:
                return data
            return None
        except json.JSONDecodeError:
            # Fallback: tentar limpar crases de markdown se houver (```json ... ```)
            try:
                cleaned = re.sub(r'```json|```', '', response_text).strip()
                # Busca novamente na string limpa
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    if "tool" in data: return data
            except:
                pass
            return None
        except Exception:
            return None

    def run_tool(self, tool_data):
        name = tool_data["tool"]
        args = tool_data["args"]
        
        print(f"⚙️ Executando ferramenta: {name}...")
        
        # Intercepta write_file para Review
        if name == "write_file":
            content = args.get("content")
            filepath = args.get("filepath")
            
            print(f"🕵️‍♂️ Solicitando revisão para '{filepath}'...")
            review_result = self.reviewer.review(content, context=f"User asked to create/edit {filepath}")
            
            if review_result.get("status") == "REJECTED":
                feedback = review_result.get("feedback", "Sem detalhes.")
                print(f"🛑 REJEITADO pelo Reviewer: {feedback}")
                # Retorna erro simulado para o modelo tentar de novo
                return f"❌ BLOQUEADO PELO REVIEWER. Motivo: {feedback}. Corrija o código e tente salvar novamente."
            else:
                print("✅ APROVADO pelo Reviewer.")
                # Prossegue para salvar
                return self.tools.write_file(filepath, content)

        if name == "execute_shell":
            return self.tools.execute_shell(args.get("command"))
        elif name == "read_file":
            return self.tools.read_file(args.get("filepath"))
        elif name == "list_files":
            return self.tools.list_files(args.get("path", "."))
        elif name == "run_python":
            return self.tools.run_python(args.get("code"))
        else:
            return f"❌ Ferramenta desconhecida: {name}"

    def _manage_memory(self):
        """Mantém o histórico limpo para não estourar tokens."""
        max_ctx = self.cfg.get("max_context", 20)
        if len(self.history) > max_ctx:
            # Mantém sempre o System Prompt (índice 0)
            # E pega as últimas (max_ctx - 1) mensagens
            self.history = [self.history[0]] + self.history[-(max_ctx-1):]

    def chat_loop(self, user_input):
        """Ciclo principal de raciocínio (ReAct simplificado)."""
        
        # 1. Adiciona input do usuário
        self.history.append({"role": "user", "content": user_input})
        
        # Limite de interações no loop para evitar loops infinitos
        max_turns = 5 
        turn = 0
        
        final_response = ""

        while turn < max_turns:
            # 2. Chama o modelo
            response = self._chat(self.history)
            
            # 3. Verifica se é tool call
            tool_data = self.process_tool_call(response)
            
            if tool_data:
                # É uma ferramenta -> Executa
                tool_result = self.run_tool(tool_data)
                
                # Adiciona a "conversa" da ferramenta no histórico
                self.history.append({"role": "assistant", "content": response})
                self.history.append({"role": "system", "content": f"TOOL_RESULT ({tool_data['tool']}): {tool_result}"})
                
                print(f"🔍 Resultado: {str(tool_result)[:200]}...") # Preview
                turn += 1
            else:
                # Resposta final (texto)
                final_response = response
                self.history.append({"role": "assistant", "content": final_response})
                break
        
        # Gerencia a memória ao fim do ciclo
        self._manage_memory()
        
        return final_response
