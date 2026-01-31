import librosa
import numpy as np
import warnings

# Matikan warning agar terminal bersih
warnings.filterwarnings('ignore')

def get_stats(feature_array, name):
    """Helper hitung Mean & Variance"""
    if feature_array.ndim > 1:
        means = np.mean(feature_array, axis=1)
        vars = np.var(feature_array, axis=1)
        result = {}
        for i in range(len(means)):
            result[f'{name}_mean_{i+1}'] = means[i]
            result[f'{name}_var_{i+1}'] = vars[i]
        return result
    else:
        return {
            f'{name}_mean': np.mean(feature_array),
            f'{name}_var': np.var(feature_array)
        }

def extract_features_from_file(file_path, duration=30):
    """
    Ekstrak fitur audio menggunakan Librosa.
    Wajib konsisten: Durasi 30 detik, SR 22050.
    """
    try:
        y, sr = librosa.load(file_path, duration=duration, sr=22050)
        
        # Jika file terlalu pendek (kurang dari 1 detik), skip
        if len(y) < sr:
            return None

        features = {}

        # 1. HPSS (Harmonic Percussive)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.update(get_stats(librosa.feature.rms(y=y_harmonic), 'hpss_harm'))
        features.update(get_stats(librosa.feature.rms(y=y_percussive), 'hpss_perc'))

        # 2. Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        features['tempo'] = tempo[0] if isinstance(tempo, np.ndarray) else tempo

        # 3. Spectral Features
        features.update(get_stats(librosa.feature.spectral_centroid(y=y, sr=sr), 'spectral_centroid'))
        features.update(get_stats(onset_env, 'spectral_flux'))
        features.update(get_stats(librosa.feature.zero_crossing_rate(y), 'zcr'))

        # 4. Chroma & MFCC
        features.update(get_stats(librosa.feature.chroma_stft(y=y_harmonic, sr=sr), 'chroma'))
        features.update(get_stats(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), 'mfcc'))

        return features

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None