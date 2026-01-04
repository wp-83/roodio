import numpy as np
import librosa
from app import config as cfg

def normalize_audio_output(val, min_val=-0.8, max_val=0.8):
    norm = (val - min_val) / (max_val - min_val)
    return float(np.clip(norm, 0.0, 1.0))

def preprocess_spectrogram(y):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.SAMPLE_RATE,
        n_mels=cfg.N_MELS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        fmax=cfg.FMAX
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    min_val, max_val = S_dB.min(), S_dB.max()
    S_norm = (S_dB - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(S_dB)

    if S_norm.shape[1] < 128:
        S_norm = np.pad(S_norm, ((0,0),(0,128-S_norm.shape[1])))
    else:
        S_norm = S_norm[:, :128]

    return S_norm[np.newaxis, ..., np.newaxis]
