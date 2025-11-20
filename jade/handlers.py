from google.colab import files
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class TextHandler:
    def process(self):
        return input("⌨️ Digite sua mensagem: ").strip()

class AudioHandler:
    def __init__(self, client, audio_model):
        self.client = client
        self.audio_model = audio_model

    def process(self):
        print("\n📤 Faça upload do áudio:")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("Nenhum arquivo enviado.")
        path = next(iter(uploaded))
        print(f"✔️ Processando '{path}'...")
        with open(path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=(path, f.read()), model=self.audio_model
            )
        text = (transcription.text or "").strip()
        print(f"🗣️ Texto: {text}")
        return text

class ImageHandler:
    def __init__(self, model_name):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process(self):
        print("\n🖼️ Faça upload da imagem:")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("Nenhuma imagem enviada.")
        path = next(iter(uploaded))
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(img, "a photo of", return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=60)
            desc = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"📋 Descrição (BLIP): {desc}")
        question = input("💬 O que deseja perguntar sobre a imagem?: ").strip()
        return question, desc
