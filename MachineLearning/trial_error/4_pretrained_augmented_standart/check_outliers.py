import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Matikan warning agar terminal bersih
warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
DATA_DIR = 'data/split_exp4/train' # Folder Training
TARGET_SR = 16000

def extract_quality_features(file_path):
    try:
        # Load audio (batasi 30 detik agar cepat, cukup untuk cek kualitas)
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=30)
        
        # 1. Energy (RMS)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 2. Brightness (Centroid)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 3. Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Librosa mengembalikan (tempo, beats), kita ambil tempo saja [0]
        tempo_output = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        
        # --- PERBAIKAN UTAMA DI SINI ---
        # Cek apakah tempo berupa array numpy? Jika ya, ambil nilainya.
        if isinstance(tempo_output, np.ndarray):
            tempo = tempo_output.item() # Mengubah array([120.]) menjadi 120.0
        else:
            tempo = tempo_output
            
        return rms, centroid, tempo
    except Exception as e:
        print(f"\n⚠️ Gagal baca {os.path.basename(file_path)}: {e}")
        return None, None, None

def detect_outliers():
    print("🕵️‍♂️ SEDANG MENGECEK KUALITAS DATA (KHUSUS .WAV)...")
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    suspicious_files = []

    for mood in moods:
        mood_dir = os.path.join(DATA_DIR, mood)
        if not os.path.exists(mood_dir): continue
        
        # KEMBALI KE VALIDASI ANDA: Hanya ambil .wav
        files = [f for f in os.listdir(mood_dir) if f.endswith('.wav')]
        
        if len(files) == 0:
            continue
            
        mood_data = []
        
        print(f"\n📂 Menganalisis Folder: {mood.upper()} ({len(files)} lagu)")
        for f in tqdm(files):
            rms, centroid, tempo = extract_quality_features(os.path.join(mood_dir, f))
            if rms is not None:
                mood_data.append({
                    'file': f,
                    'rms': rms,
                    'centroid': centroid,
                    'tempo': tempo
                })
        
        if not mood_data: continue

        # Hitung statistik rata-rata folder
        df = pd.DataFrame(mood_data)
        mean_rms = df['rms'].mean()
        std_rms = df['rms'].std()
        mean_cen = df['centroid'].mean()
        std_cen = df['centroid'].std()
        
        # Deteksi Anomali
        for index, row in df.iterrows():
            reasons = []
            
            # 1. Cek Loudness (RMS)
            if row['rms'] > mean_rms + 1.5 * std_rms:
                reasons.append("Terlalu Berisik")
            elif row['rms'] < mean_rms - 1.5 * std_rms:
                reasons.append("Terlalu Pelan")
                
            # 2. Cek Brightness (Centroid)
            if row['centroid'] > mean_cen + 1.5 * std_cen:
                reasons.append("Terlalu Cempreng")
            elif row['centroid'] < mean_cen - 1.5 * std_cen:
                reasons.append("Terlalu Mendem")
                
            # 3. Cek Tempo (Spesifik Mood)
            # Karena row['tempo'] sudah dipastikan float di fungsi extract, 
            # error formatting tidak akan muncul lagi.
            if (mood in ['sad', 'relaxed']) and (row['tempo'] > 130):
                reasons.append(f"Tempo Terlalu Cepat ({row['tempo']:.0f} BPM)")
            
            if (mood in ['happy', 'angry']) and (row['tempo'] < 80):
                reasons.append(f"Tempo Terlalu Lambat ({row['tempo']:.0f} BPM)")

            if reasons:
                suspicious_files.append({
                    'Mood': mood,
                    'File': row['file'],
                    'Masalah': ", ".join(reasons)
                })

    # Laporan Akhir
    print("\n" + "="*60)
    print("🚨 LAPORAN LAGU BERMASALAH (HIGH VARIANCE) 🚨")
    print("="*60)
    
    if len(suspicious_files) == 0:
        print("✅ Data WAV terlihat seragam. Bagus!")
    else:
        df_suspect = pd.DataFrame(suspicious_files)
        for mood in moods:
            subset = df_suspect[df_suspect['Mood'] == mood]
            if not subset.empty:
                print(f"\n📁 {mood.upper()} ({len(subset)} suspect):")
                for _, row in subset.iterrows():
                    print(f"  ❌ {row['File']}")
                    print(f"     └-> {row['Masalah']}")

if __name__ == "__main__":
    detect_outliers()