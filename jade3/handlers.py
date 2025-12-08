from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# TextHandler e AudioHandler podem ser removidos se você SÓ usa a interface Gradio,
# mas vou manter para não quebrar nada se você tiver outro uso para eles.

class TextHandler:
    def process(self):
        return input("⌨️ Digite sua mensagem: ").strip()

class AudioHandler:
    def __init__(self, client, audio_model):
        self.client = client
        self.audio_model = audio_model
    # ... (código do AudioHandler original pode ficar aqui)



class ImageHandler:
    def __init__(self, model_name):
        # CORREÇÃO AQUI: Adicionado 'use_fast=True' para silenciar o aviso.
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process_pil_image(self, pil_image: Image.Image):
        """Processa um objeto PIL.Image vindo diretamente do Gradio."""
        if not isinstance(pil_image, Image.Image):
            raise TypeError("A entrada deve ser um objeto PIL.Image.")
        return self._generate_caption(pil_image.convert("RGB"))

    def _generate_caption(self, img):
        """Lógica de geração de legenda reutilizável."""
        with torch.no_grad():
            inputs = self.processor(img, "a photo of", return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=60)
            return self.processor.decode(out[0], skip_special_tokens=True).strip()
