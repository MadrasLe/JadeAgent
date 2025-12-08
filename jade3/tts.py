from gtts import gTTS
from IPython.display import HTML, display
import base64, io

class TTSPlayer:
    def __init__(self, lang="pt"):
        self.lang = lang

    def play(self, text):
        tts = gTTS(text, lang=self.lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_bytes = buf.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f'<audio src="data:audio/mpeg;base64,{audio_b64}" controls autoplay></audio>'
        display(HTML(audio_html))
