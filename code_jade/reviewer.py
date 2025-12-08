import json
from groq import Groq

class CodeReviewer:
    def __init__(self, config):
        self.client = Groq(api_key=self._get_api_key())
        self.model = config.get("groq_model", "llama3-70b-8192")
        
        self.system_prompt = """Você é um Code Reviewer Sênior, rigoroso e focado em Segurança e Qualidade.
Sua função é analisar trechos de código e aprovar ou rejeitar baseando-se em:
1. Bugs lógicos óbvios.
2. Riscos de Segurança (ex: SQL Injection, exec sem validação, hardcoded credentials).
3. Boas práticas (variáveis legíveis, tratamento de erro básico).

FORMATO DE RESPOSTA (JSON OBRIGATÓRIO):
Você deve responder APENAS um JSON no seguinte formato:
{
  "status": "APPROVED" | "REJECTED",
  "feedback": "Explicação curta do problema (se REJECTED) ou elogio (se APPROVED)."
}

Se o código for seguro e funcional, aprove. Não seja pedante com estilo (PEP8) a menos que torne o código ilegível.
Se houver perigo real ou erro fatal, rejeite.
"""

    def _get_api_key(self):
        import os
        key = os.getenv("GROQ_API_KEY")
        if not key:
             try:
                from google.colab import userdata
                key = userdata.get('GROQ_API_KEY')
             except:
                pass
        return key or "dummy"

    def review(self, code, context=""):
        """Analisa o código e retorna status e feedback."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Contexto: {context}\n\nCódigo para revisão:\n```python\n{code}\n```"}
        ]
        
        try:
            chat = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.1, # Baixa temperatura para análise fria
                response_format={"type": "json_object"} # Força JSON mode no Llama 3 se disponível, ou ajuda
            )
            response_text = chat.choices[0].message.content
            
            # Tenta parsear
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback simples se não vier JSON limpo
                if "APPROVED" in response_text.upper():
                    return {"status": "APPROVED", "feedback": "Parser failed but looks approved."}
                return {"status": "REJECTED", "feedback": f"Parser failed. Raw response: {response_text}"}
                
        except Exception as e:
            return {"status": "ERROR", "feedback": str(e)}
