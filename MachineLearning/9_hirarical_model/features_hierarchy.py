import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet Global (Sekali saja biar cepat)
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_yamnet_embedding(wav):
    # Preprocessing standar YAMNet
    wav = wav / np.max(np.abs(wav))
    if len(wav) < 16000: wav = np.pad(wav, (0, 16000 - len(wav)))
    _, embeddings, _ = yamnet_model(wav)
    
    # Pooling: Mean + Std + Max
    mean = tf.reduce_mean(embeddings, axis=0)
    std = tf.math.reduce_std(embeddings, axis=0)
    max_ = tf.reduce_max(embeddings, axis=0)
    return tf.concat([mean, std, max_], axis=0).numpy()

# --- STAGE 1: AROUSAL SPECIALIST ---
# Fokus: Fisika Suara (Energi & Tempo)
def extract_stage_1(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    yamnet_emb = get_yamnet_embedding(y)
    
    # RMS (Kekerasan) & ZCR (Kekasaran/Noise)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # Output: 3072 + 2 = 3074 dimensi
    return np.concatenate([yamnet_emb, [rms, zcr]])

# --- STAGE 2A: HIGH AROUSAL VALENCE (Angry vs Happy) ---
# Fokus: Keteraturan (Happy) vs Distorsi/Noise (Angry)
def extract_stage_2a(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    yamnet_emb = get_yamnet_embedding(y)
    
    # Spectral Contrast (Bagus membedakan Harmonic vs Noise)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1) # 7 bands
    
    # Spectral Flatness (Flat = Noise/Angry)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    return np.concatenate([yamnet_emb, contrast_mean, [flatness]])

# --- STAGE 2B: LOW AROUSAL VALENCE (Sad vs Relaxed) ---
# Fokus: Warna Nada (Major/Airy) vs (Minor/Dark)
def extract_stage_2b(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y_harmonic = librosa.effects.harmonic(y) # Ambil harmonik saja
    
    yamnet_emb = get_yamnet_embedding(y)
    
    # 1. Tonnetz (Mood Harmoni)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # 2. Spectral Roll-off (Kecerahan)
    # Sad = Dark (Low freq), Relaxed = Bright (High freq)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    rolloff_norm = rolloff / 1000.0 # Normalisasi sederhana
    
    return np.concatenate([yamnet_emb, tonnetz_mean, [rolloff_norm]])