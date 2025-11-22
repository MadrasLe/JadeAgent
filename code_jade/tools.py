import os
import subprocess
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

class ToolManager:
    def __init__(self, safe_mode=True, work_dir="."):
        self.safe_mode = safe_mode
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

    def _confirm(self, action, content):
        if not self.safe_mode:
            return True
        print(f"\n⚠️  [SOLICITAÇÃO DE EXECUÇÃO]")
        print(f"Ação: {action}")
        print(f"Detalhes: {content}")
        resp = input(">> Permitir? (s/n): ").strip().lower()
        return resp == "s"

    def execute_shell(self, command):
        """Executa comandos no terminal."""
        if not self._confirm("SHELL", command):
            return "❌ Ação negada pelo usuário."
        
        try:
            # Executa no diretório de trabalho
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=self.work_dir, 
                text=True, 
                capture_output=True,
                timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            return output.strip() or "[Comando executado sem saída]"
        except Exception as e:
            return f"❌ Erro de execução: {str(e)}"

    def write_file(self, filepath, content):
        """Cria ou sobrescreve um arquivo."""
        full_path = os.path.join(self.work_dir, filepath)
        
        if not self._confirm("WRITE_FILE", f"{filepath} ({len(content)} chars)"):
            return "❌ Ação negada pelo usuário."

        try:
            # Cria diretórios pai se não existirem
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Arquivo '{filepath}' salvo com sucesso."
        except Exception as e:
            return f"❌ Erro ao salvar arquivo: {str(e)}"

    def read_file(self, filepath):
        """Lê o conteúdo de um arquivo."""
        full_path = os.path.join(self.work_dir, filepath)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"❌ Arquivo não encontrado: {filepath}"
        except Exception as e:
            return f"❌ Erro ao ler arquivo: {str(e)}"

    def list_files(self, path="."):
        """Lista arquivos no diretório."""
        # Se path for '.', usa o diretório atual (que já é o work_dir no execute_shell)
        # Se path for algo mais, concatena
        try:
            cmd = f"ls -R -F {path}"
            return self.execute_shell(cmd)
        except Exception as e:
            return f"❌ Erro ao listar arquivos: {str(e)}"

    def run_python(self, code):
        """Executa código Python e captura a saída."""
        if not self._confirm("PYTHON_EXEC", code[:200] + "..."):
            return "❌ Ação negada pelo usuário."

        # Captura stdout e stderr
        f_out = io.StringIO()
        f_err = io.StringIO()
        
        # Sandbox simples: define globais restritos
        # IMPORTANTE: exec() nunca é 100% seguro sem containers reais.
        safe_globals = {
            "__builtins__": __builtins__, 
            "__name__": "__main__",
            "math": __import__("math"),
            "json": __import__("json"),
            "os": __import__("os"), # CodeJade precisa de OS muitas vezes, mas é perigoso.
            # Usuário confia no agente, mas evitamos acesso ao `self` do ToolManager
        }

        try:
            with redirect_stdout(f_out), redirect_stderr(f_err):
                exec(code, safe_globals)
            
            output = f_out.getvalue()
            errors = f_err.getvalue()
            
            res = ""
            if output: res += f"[STDOUT]\n{output}\n"
            if errors: res += f"[STDERR]\n{errors}\n"
            
            return res.strip() or "[Código executado sem saída]"
        except Exception as e:
            return f"❌ Erro de execução Python: {e}"
