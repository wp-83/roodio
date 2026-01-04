from fastapi import FastAPI, Query
import os
import numpy as np
import librosa
import tempfile
import requests

from app import config as cfg
from app.deps import load_models
from app.audio_service import preprocess_spectrogram, normalize_audio_output
from app.fusion_service import get_mood_label
from app.schemas import EmotionResponse

app = FastAPI(title="Emotion Analyzer API")

# ======================================================
# Load models sekali (singleton)
# ======================================================
audio_model, lyrics_model = load_models()


def to_float(val):
    """Pastikan numpy -> python native"""
    if isinstance(val, np.generic):
        return val.item()
    return float(val)


def download_blob_to_temp(url: str) -> str:
    """
    Jika URL Azure Blob, download sementara ke file temp.
    """
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise ValueError(f"Failed to download audio: {resp.status_code}")

    suffix = os.path.splitext(url)[-1] or ".mp3"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp_file.name, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp_file.name


@app.post("/analyze", response_model=EmotionResponse)
async def analyze_emotion(
    audio_path: str = Query(..., description="Absolute path local atau Azure Blob URL"),
    lyrics: str = Query(...),
    w_val_audio: float = Query(0.4, ge=0.0, le=1.0),
    w_aro_audio: float = Query(0.7, ge=0.0, le=1.0),
):
    tmp_file_path = None

    try:
        # --------------------------------------------------
        # Jika audio_path URL → download sementara
        # --------------------------------------------------
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            tmp_file_path = download_blob_to_temp(audio_path)
            audio_path_to_use = tmp_file_path
        else:
            audio_path_to_use = audio_path.strip().strip('"').strip("'")

        # --------------------------------------------------
        # Validasi path
        # --------------------------------------------------
        if not os.path.isfile(audio_path_to_use):
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "mood": "Unknown",
                "audio": {},
                "lyrics": {}
            }

        # --------------------------------------------------
        # Load audio
        # --------------------------------------------------
        y, _ = librosa.load(
            audio_path_to_use,
            sr=cfg.SAMPLE_RATE,
            mono=True,
            duration=cfg.DURATION,
            offset=cfg.SKIP_SECONDS
        )

        # --------------------------------------------------
        # AUDIO ANALYSIS
        # --------------------------------------------------
        spec = preprocess_spectrogram(y)
        audio_pred = audio_model.predict(spec)
        audio_pred = np.squeeze(audio_pred)

        audio_val = normalize_audio_output(audio_pred[0])
        audio_aro = normalize_audio_output(audio_pred[1])

        audio_val = to_float(audio_val)
        audio_aro = to_float(audio_aro)

        # --------------------------------------------------
        # LYRICS ANALYSIS
        # --------------------------------------------------
        lyr_val, lyr_aro, lyr_meta = lyrics_model.predict(lyrics)

        lyr_val = to_float(lyr_val)
        lyr_aro = to_float(lyr_aro)

        # --------------------------------------------------
        # FUSION
        # --------------------------------------------------
        final_val = w_val_audio * audio_val + (1 - w_val_audio) * lyr_val
        final_aro = w_aro_audio * audio_aro + (1 - w_aro_audio) * lyr_aro

        final_val = to_float(final_val)
        final_aro = to_float(final_aro)

        mood = get_mood_label(final_val, final_aro)

        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------
        return {
            "valence": final_val,
            "arousal": final_aro,
            "mood": mood,
            "audio": {
                "valence": audio_val,
                "arousal": audio_aro
            },
            "lyrics": {
                "valence": lyr_val,
                "arousal": lyr_aro,
                "keywords": lyr_meta.get("keywords", [])
            }
        }

    finally:
        # Hapus file sementara jika ada
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
