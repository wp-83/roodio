import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet Global (Supaya tidak load berulang-ulang)
print("⏳ Loading YAMNet Model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet Loaded.")

# --- FUNGSI PREPROCESSING (TRIM TENGAH) ---
def trim_middle(y, sr, percentage=0.5):
    """Mengambil 50% bagian tengah audio"""
    if len(y) < sr: return y # Jangan potong kalau kependekan (< 1 detik)
    
    duration = len(y)
    start = int(duration * (1 - percentage) / 2) # Titik mulai (25%)
    end = start + int(duration * percentage)     # Titik akhir (75%)
    
    return y[start:end]

def get_yamnet_embedding(wav):
    # Preprocessing standar YAMNet
    # 1. Normalize
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
    
    # 2. Pad jika terlalu pendek (Wajib untuk YAMNet)
    if len(wav) < 16000: 
        wav = np.pad(wav, (0, 16000 - len(wav)))
        
    _, embeddings, _ = yamnet_model(wav)
    
    # Pooling: Mean + Std + Max
    mean = tf.reduce_mean(embeddings, axis=0)
    std = tf.math.reduce_std(embeddings, axis=0)
    max_ = tf.reduce_max(embeddings, axis=0)
    return tf.concat([mean, std, max_], axis=0).numpy()

# --- STAGE 1: AROUSAL (Energy) ---
def extract_stage_1(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = trim_middle(y, sr) # POTONG TENGAH DULU
        
        yamnet_emb = get_yamnet_embedding(y)
        
        # Fitur Fisika: RMS (Keras) & ZCR (Kasar)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        return np.concatenate([yamnet_emb, [rms, zcr]])
    except:
        return None

# --- STAGE 2A: HIGH VALENCE (Angry vs Happy) ---
def extract_stage_2a(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = trim_middle(y, sr)
        
        yamnet_emb = get_yamnet_embedding(y)
        
        # Fitur: Spectral Contrast (Harmoni vs Noise)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # Fitur: Spectral Flatness (Noise level)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        return np.concatenate([yamnet_emb, contrast_mean, [flatness]])
    except:
        return None

# --- STAGE 2B: LOW VALENCE (Sad vs Relaxed) ---
def extract_stage_2b(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = trim_middle(y, sr)
        
        y_harmonic = librosa.effects.harmonic(y) # Ambil harmonik
        yamnet_emb = get_yamnet_embedding(y)
        
        # Fitur: Tonnetz (Mayor/Minor)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Fitur: Rolloff (Bright/Dark)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        rolloff_norm = rolloff / 1000.0
        
        return np.concatenate([yamnet_emb, tonnetz_mean, [rolloff_norm]])
    except:
        return None