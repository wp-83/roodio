from fastapi import FastAPI, File, UploadFile, Form
import os
import numpy as np
import librosa
import tempfile
import shutil  # Tambahan untuk menyalin file upload ke temp

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


@app.post("/analyze", response_model=EmotionResponse)
async def analyze_emotion(
    # Menggunakan UploadFile untuk file
    audio_file: UploadFile = File(..., description="File audio (mp3/wav)"),
    # Menggunakan Form untuk data teks karena Content-Type adalah multipart/form-data
    lyrics: str = Form(...),
    w_val_audio: float = Form(0.4, ge=0.0, le=1.0),
    w_aro_audio: float = Form(0.7, ge=0.0, le=1.0),
):
    tmp_file_path = None

    try:
        # --------------------------------------------------
        # 1. Simpan file upload ke Temporary File
        # --------------------------------------------------
        # Kita perlu simpan ke disk dulu karena librosa.load dengan offset/duration
        # bekerja paling stabil dengan path file fisik.
        
        # Buat nama file temp dengan ekstensi asli (misal .mp3)
        suffix = os.path.splitext(audio_file.filename)[-1] or ".mp3"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Salin isi file yang diupload ke file temp
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_file_path = tmp.name

        # --------------------------------------------------
        # 2. Load audio dengan Librosa
        # --------------------------------------------------
        y, _ = librosa.load(
            tmp_file_path,
            sr=cfg.SAMPLE_RATE,
            mono=True,
            duration=cfg.DURATION,
            offset=cfg.SKIP_SECONDS
        )

        # --------------------------------------------------
        # 3. AUDIO ANALYSIS
        # --------------------------------------------------
        spec = preprocess_spectrogram(y)
        audio_pred = audio_model.predict(spec)
        audio_pred = np.squeeze(audio_pred)

        audio_val = normalize_audio_output(audio_pred[0])
        audio_aro = normalize_audio_output(audio_pred[1])

        audio_val = to_float(audio_val)
        audio_aro = to_float(audio_aro)

        # --------------------------------------------------
        # 4. LYRICS ANALYSIS
        # --------------------------------------------------
        lyr_val, lyr_aro, lyr_meta = lyrics_model.predict(lyrics)

        lyr_val = to_float(lyr_val)
        lyr_aro = to_float(lyr_aro)

        # --------------------------------------------------
        # 5. FUSION
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

    except Exception as e:
        # Opsional: Tangkap error agar tidak 500 Internal Server Error tanpa info
        print(f"Error processing: {e}")
        raise e

    finally:
        # --------------------------------------------------
        # Cleanup: Hapus file temp
        # --------------------------------------------------
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)