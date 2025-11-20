import json
import time
import sys
import logging
from groq import Groq
from .handlers import TextHandler, AudioHandler, ImageHandler
from .tts import TTSPlayer
from .utils import slim_history

class JadeAgent:
    def __init__(self, config_path="jade/config.json"):
        # Configuração
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.client = Groq(api_key=self._get_api_key())
        self.history = [{"role": "system", "content": "Você é J.A.D.E., uma IA multimodal calma, direta e inteligente."}]
        self.cycle = 1

        # Handlers
        self.text_handler = TextHandler()
        self.audio_handler = AudioHandler(self.client, self.cfg["audio_model"])
        self.image_handler = ImageHandler(self.cfg["caption_model"])
        self.tts = TTSPlayer(lang=self.cfg.get("language", "pt"))

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("J.A.D.E. iniciada com sucesso.")

    def _get_api_key(self):
        import os
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("❌ GROQ_API_KEY não encontrada. Defina via os.environ ou .env")
        return key

    def _chat(self, messages):
        """Chama o modelo Groq Llama."""
        chat = self.client.chat.completions.create(
            messages=messages,
            model=self.cfg["groq_model"]
        )
        return chat.choices[0].message.content.strip()

    def respond(self, user_input, vision_context=None):
        """Raciocínio principal + síntese"""
        self.history.append({"role": "user", "content": user_input})
        messages = self.history[:]
        if vision_context:
            messages.append({"role": "system", "content": f"Contexto visual: {vision_context}"})

        resposta = self._chat(messages)
        self.history.append({"role": "assistant", "content": resposta})
        self.history = slim_history(self.history, keep=self.cfg["max_context"])

        print(f"\n🤖 J.A.D.E.: {resposta}")
        try:
            self.tts.play(resposta)
        except Exception as e:
            logging.warning(f"TTS falhou: {e}")
        return resposta

    def run(self):
        """Loop principal"""
        while True:
            try:
                print("\n--------------------------------------------------")
                print(f"🌀 [CICLO {self.cycle}] J.A.D.E. ativa.")
                modo = input("🎤 Escolha: texto, audio, imagem ou sair → ").lower().strip()

                if modo == "sair":
                    print("👋 Encerrando sessão da J.A.D.E.")
                    break

                if modo == "texto":
                    user_input = self.text_handler.process()
                    self.respond(user_input)

                elif modo == "audio":
                    user_input = self.audio_handler.process()
                    self.respond(user_input)

                elif modo == "imagem":
                    user_input, vision_context = self.image_handler.process()
                    self.respond(user_input, vision_context)

                else:
                    print("⚠️ Opção inválida.")
                    continue

                self.cycle += 1
                sys.stdout.flush()
                time.sleep(1.2)

            except Exception as e:
                logging.error(f"Erro no ciclo {self.cycle}: {e}")
                if self.history and self.history[-1]["role"] == "user":
                    self.history.pop()
                time.sleep(3)
                continue
