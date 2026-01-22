import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Abaikan warning jika ada file audio yang formatnya aneh sedikit
warnings.filterwarnings('ignore')

# --- KONFIGURASI PATH (Sama seperti sebelumnya) ---
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'interim')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'features.csv')

def get_stats(feature_array, name):
    """
    Helper function untuk mengambil Mean dan Variance.
    Mengembalikan dictionary agar nama kolom otomatis rapi.
    """
    # Jika feature array 2D (misal MFCC: n_mfcc x time), ratakan per baris
    if feature_array.ndim > 1:
        means = np.mean(feature_array, axis=1)
        vars = np.var(feature_array, axis=1)
        
        result = {}
        for i in range(len(means)):
            result[f'{name}_mean_{i+1}'] = means[i]
            result[f'{name}_var_{i+1}'] = vars[i]
        return result
    
    # Jika feature array 1D (misal ZCR), langsung hitung scalar
    else:
        return {
            f'{name}_mean': np.mean(feature_array),
            f'{name}_var': np.var(feature_array)
        }

def extract_features(file_path):
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, duration=30)
        
        features = {}

        # 2. HPSS (Harmonic-Percussive Source Separation)
        # Memisahkan komponen harmoni (melodi) dan perkusi (beat)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Ambil RMS (Energi) dari komponen HPSS
        rms_harmonic = librosa.feature.rms(y=y_harmonic)
        rms_percussive = librosa.feature.rms(y=y_percussive)
        
        features.update(get_stats(rms_harmonic, 'hpss_harm'))
        features.update(get_stats(rms_percussive, 'hpss_perc'))

        # 3. Tempo (BPM)
        # Tempo biasanya single value (scalar), jadi tidak butuh variance
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        # Librosa baru mengembalikan array, kita ambil nilai pertamanya
        features['tempo'] = tempo[0] if isinstance(tempo, np.ndarray) else tempo

        # 4. Spectral Centroid (Kecerahan Suara)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.update(get_stats(cent, 'spectral_centroid'))

        # 5. Spectral Flux (Tingkat Perubahan / Onset Strength)
        # Kita gunakan onset_env yang sudah dihitung di poin 3
        features.update(get_stats(onset_env, 'spectral_flux'))

        # 6. Zero Crossing Rate (ZCR) - Kebisingan/Noisiness
        zcr = librosa.feature.zero_crossing_rate(y)
        features.update(get_stats(zcr, 'zcr'))

        # 7. Chroma (Kunci Nada) - Gunakan STFT
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        features.update(get_stats(chroma, 'chroma'))

        # 8. MFCC (Warna Suara) - Menggunakan 20 koefisien standar
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.update(get_stats(mfcc, 'mfcc'))

        return features

    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def process_data():
    print(f"Membaca data dari: {INPUT_PATH}")
    
    all_rows = []
    
    if not os.path.exists(INPUT_PATH):
        print("Folder data/interim tidak ditemukan!")
        return

    moods = os.listdir(INPUT_PATH)
    
    for mood in moods:
        mood_folder = os.path.join(INPUT_PATH, mood)
        if not os.path.isdir(mood_folder):
            continue
            
        print(f"Mengekstrak fitur (Advanced) untuk mood: {mood}...")
        
        for filename in os.listdir(mood_folder):
            file_path = os.path.join(mood_folder, filename)
            
            # Ekstrak dictionary fitur
            feature_dict = extract_features(file_path)
            
            if feature_dict:
                # Tambahkan label
                feature_dict['label'] = mood
                all_rows.append(feature_dict)

    # Convert ke DataFrame
    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)
        
        # Pindahkan kolom 'label' ke paling kanan agar rapi
        cols = [c for c in df.columns if c != 'label'] + ['label']
        df = df[cols]
        
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*40)
        print(f"SUKSES! Data tersimpan di: {OUTPUT_FILE}")
        print(f"Total Sampel: {len(df)}")
        print(f"Total Fitur per sampel: {len(cols)-1}") # dikurangi label
        print("="*40)
    else:
        print("Tidak ada data yang berhasil diekstrak.")

if __name__ == "__main__":
    process_data()