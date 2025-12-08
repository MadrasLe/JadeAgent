import json
import logging
import os
import sys
import time
import uuid

from groq import Groq

# Importa nossos módulos customizados
from .handlers import ImageHandler
from .tts import TTSPlayer
from .utils import slim_history
from .shorestone import ShoreStoneMemory
from .curator_heuristic import MemoryCuratorHeuristic

# Configura o logger principal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - JADE - %(levelname)s - %(message)s")

class JadeAgent:
    def __init__(self, config_path="jade/config.json"):
        # Carrega configurações
        with open(config_path) as f:
            self.cfg = json.load(f)

        # --- Configuração da API Groq ---
        logging.info("Iniciando J.A.D.E. em modo API (Groq)...")
        self.api_key = self._get_api_key()
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.cfg.get("groq_model", "llama3-8b-instant")

        # Histórico base
        self.history = [{"role": "system", "content": "Você é J.A.D.E., uma IA multimodal calma e inteligente. Seja direta. Responda de forma concisa e natural. NÃO explique seu processo de pensamento. Apenas responda à pergunta."}]
        
        # --- Inicialização dos Módulos ---
        logging.info("Carregando módulos de percepção e memória...")
        
        # Visão e Fala
        self.image_handler = ImageHandler(self.cfg.get("caption_model", "Salesforce/blip-image-captioning-large"))
        self.tts = TTSPlayer(lang=self.cfg.get("language", "pt"))
        
        # 1. Memória ShoreStone (Persistente)
        self.memory = ShoreStoneMemory()
        self.memory.load_or_create_session("sessao_padrao_gabriel")
        
        # 2. Curador Heurístico (Manutenção Automática)
        self.curator = MemoryCuratorHeuristic(shorestone_memory=self.memory)
        self.response_count = 0
        self.maintenance_interval = 10 # Executar a manutenção a cada 10 interações

        logging.info(f"J.A.D.E. pronta e conectada ao modelo {self.model_name}.")

    def _get_api_key(self):
        """Recupera a chave da API do ambiente de forma segura."""
        key = os.getenv("GROQ_API_KEY")
        if not key:
            logging.error("Chave GROQ_API_KEY não encontrada nas variáveis de ambiente.")
            raise RuntimeError("❌ GROQ_API_KEY não encontrada. Defina a variável de ambiente.")
        return key

    def _chat(self, messages):
        """Envia as mensagens para a Groq e retorna a resposta."""
        try:
            chat = self.client.chat.completions.create(
                messages=messages, 
                model=self.model_name,
                temperature=0.7, # Criatividade balanceada
                max_tokens=1024  # Limite de resposta razoável
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Erro na comunicação com a Groq: {e}")
            return "Desculpe, tive um problema ao me conectar com meu cérebro na nuvem."

    def respond(self, user_input, vision_context=None):
        """Processo principal de raciocínio: Lembrar -> Ver -> Responder -> Memorizar -> Manter."""
        
        messages = self.history[:]
        
        # 1. Lembrar (Recuperação de Contexto)
        memories = self.memory.remember(user_input)
        if memories:
            memory_context = f"--- MEMÓRIAS RELEVANTES (ShoreStone) ---\n{memories}\n--- FIM DAS MEMÓRIAS ---"
            # Inserimos as memórias como contexto de sistema para guiar a resposta
            messages.append({"role": "system", "content": memory_context})

        # 2. Ver (Contexto Visual)
        if vision_context:
            messages.append({"role": "system", "content": f"Contexto visual da imagem que o usuário enviou: {vision_context}"})

        # Adiciona a pergunta atual ao histórico temporário e ao prompt
        self.history.append({"role": "user", "content": user_input})
        messages.append({"role": "user", "content": user_input})

        # 3. Responder (Geração)
        resposta = self._chat(messages)
        
        # Atualiza histórico
        self.history.append({"role": "assistant", "content": resposta})
        self.history = slim_history(self.history, keep=self.cfg.get("max_context", 12))
        
        # 4. Memorizar (Armazenamento Persistente)
        self.memory.memorize(user_input, resposta)

        print(f"\n🤖 J.A.D.E.: {resposta}")
        
        # Falar (TTS)
        try:
            self.tts.play(resposta)
        except Exception as e:
            logging.warning(f"TTS falhou (silenciado): {e}")
            
        # 5. Manter (Ciclo de Curadoria Automática)
        self.response_count += 1
        if self.response_count % self.maintenance_interval == 0:
            logging.info(f"Ciclo de manutenção agendado (interação {self.response_count}). Verificando saúde da memória...")
            try:
                self.curator.run_maintenance_cycle()
            except Exception as e:
                logging.error(f"Erro no Curador de Memória: {e}")
        
        return resposta