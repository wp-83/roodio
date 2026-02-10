import librosa
import numpy as np
import warnings

# Matikan warning agar terminal bersih
warnings.filterwarnings('ignore')

def get_stats(feature_array, name):
    """
    UPGRADE: Menghitung Mean, Variance, DAN Range.
    Range (Max-Min) membantu mendeteksi dinamika emosi.
    """
    if feature_array.ndim > 1:
        means = np.mean(feature_array, axis=1)
        vars = np.var(feature_array, axis=1)
        ranges = np.ptp(feature_array, axis=1) # Peak-to-Peak (Max - Min)
        
        result = {}
        for i in range(len(means)):
            result[f'{name}_mean_{i+1}'] = means[i]
            result[f'{name}_var_{i+1}'] = vars[i]
            result[f'{name}_range_{i+1}'] = ranges[i] # Fitur Baru
        return result
    else:
        return {
            f'{name}_mean': np.mean(feature_array),
            f'{name}_var': np.var(feature_array),
            f'{name}_range': np.ptp(feature_array) # Fitur Baru
        }

def extract_features_from_file(file_path, duration=30):
    """
    Ekstrak fitur audio ADVANCED (MFCC + Tonnetz + Contrast + Spectral).
    """
    try:
        # Load Audio
        y, sr = librosa.load(file_path, duration=duration, sr=22050)
        
        if len(y) < sr:
            return None

        features = {}

        # 1. HPSS (Harmonic Percussive)
        y_harm, y_perc = librosa.effects.hpss(y)
        features.update(get_stats(librosa.feature.rms(y=y_harm), 'hpss_harm'))
        features.update(get_stats(librosa.feature.rms(y=y_perc), 'hpss_perc'))

        # 2. Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        features['tempo'] = tempo[0] if isinstance(tempo, np.ndarray) else tempo

        # 3. Spectral Features (DITAMBAH BANYAK)
        features.update(get_stats(librosa.feature.spectral_centroid(y=y, sr=sr), 'spectral_centroid'))
        
        # [BARU] Bandwidth: Lebar frekuensi (Terang vs Redup)
        features.update(get_stats(librosa.feature.spectral_bandwidth(y=y, sr=sr), 'spectral_bandwidth')) 
        
        # [BARU] Rolloff: Batas frekuensi tinggi (Kecerahan suara)
        features.update(get_stats(librosa.feature.spectral_rolloff(y=y, sr=sr), 'spectral_rolloff'))     
        
        # [BARU & PENTING] Contrast: Membedakan tekstur bersih vs kasar (Happy vs Angry)
        features.update(get_stats(librosa.feature.spectral_contrast(y=y_harm, sr=sr), 'spectral_contrast')) 

        features.update(get_stats(onset_env, 'spectral_flux'))
        features.update(get_stats(librosa.feature.zero_crossing_rate(y), 'zcr'))

        # 4. Chroma & Tonnetz
        features.update(get_stats(librosa.feature.chroma_stft(y=y_harm, sr=sr), 'chroma'))
        
        # [BARU] Tonnetz: Deteksi harmoni Mayor/Minor (Happy vs Sad)
        features.update(get_stats(librosa.feature.tonnetz(y=y_harm, sr=sr), 'tonnetz')) 

        # 5. MFCC (Tetap 20)
        features.update(get_stats(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), 'mfcc'))

        return features

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None