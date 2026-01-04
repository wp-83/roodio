import os
from tensorflow.keras.models import load_model
from app import config as cfg
from app.lyrics_analyzer import NRCLexiconAnalyzer

audio_model = None
lyrics_model = None

def load_models():
    global audio_model, lyrics_model

    if audio_model is None:
        audio_path = os.path.join(cfg.MODELS_DIR, "model_emosi_cnn.keras")
        audio_model = load_model(audio_path)

    if lyrics_model is None:
        lyrics_model = NRCLexiconAnalyzer()

    return audio_model, lyrics_model
