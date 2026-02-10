import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- KONFIGURASI PATH ---
DEAM_AUDIO_DIR = r'./data/deam/audio' 
CSV_PATH_1 = r'./data/deam/annotations/static_annotations_averaged_songs_1_2000.csv'
CSV_PATH_2 = r'./data/deam/annotations/static_annotations_averaged_songs_2000_2058.csv'

OUTPUT_DIR = 'data/training'

def prepare_deam_data():
    print("📂 Menyiapkan Data DEAM (Dengan Quality Control STD)...")
    
    # 1. Cek & Load CSV
    try:
        df1 = pd.read_csv(CSV_PATH_1)
        df2 = pd.read_csv(CSV_PATH_2)
        df = pd.concat([df1, df2], ignore_index=True)
        df.columns = df.columns.str.strip() # Hapus spasi di nama kolom
        print(f"✅ CSV Tergabung! Total Baris: {len(df)}")
    except Exception as e:
        print(f"❌ Error Load CSV: {e}")
        return

    # 2. Reset Folder Output
    moods = ['happy', 'sad', 'angry', 'relaxed']
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for mood in moods:
        os.makedirs(os.path.join(OUTPUT_DIR, mood), exist_ok=True)

    # 3. FILTERING LOGIC
    print("🚀 Menyaring Audio...")
    
    count = {'happy': 0, 'sad': 0, 'angry': 0, 'relaxed': 0}
    skipped = {'ambigu': 0, 'high_std': 0, 'missing_file': 0}
    
    # PARAMETER FILTER
    BUFFER = 0.25      # Zona aman dari titik tengah (5.0)
    MAX_STD = 2.5      # Batas toleransi perdebatan annotator (Semakin kecil semakin ketat)
                       # 2.5 adalah angka moderat untuk dataset emosi subjektif.
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            song_id = int(row['song_id'])
            
            v_mean = float(row['valence_mean'])
            a_mean = float(row['arousal_mean'])
            v_std  = float(row['valence_std'])
            a_std  = float(row['arousal_std'])
            
            # --- FILTER 1: CONSISTENCY CHECK (STD) ---
            # Jika annotator terlalu banyak berdebat, buang lagunya.
            if v_std > MAX_STD or a_std > MAX_STD:
                skipped['high_std'] += 1
                continue 

            # --- FILTER 2: QUADRANT MAPPING (MEAN) ---
            target_mood = None
            
            # Happy: Valence Tinggi, Arousal Tinggi
            if v_mean > (5 + BUFFER) and a_mean > (5 + BUFFER):
                target_mood = 'happy'
            # Angry: Valence Rendah, Arousal Tinggi
            elif v_mean < (5 - BUFFER) and a_mean > (5 + BUFFER):
                target_mood = 'angry'
            # Sad: Valence Rendah, Arousal Rendah
            elif v_mean < (5 - BUFFER) and a_mean < (5 - BUFFER):
                target_mood = 'sad'
            # Relaxed: Valence Tinggi, Arousal Rendah
            elif v_mean > (5 + BUFFER) and a_mean < (5 - BUFFER):
                target_mood = 'relaxed'
            else:
                skipped['ambigu'] += 1
                continue 

            # --- COPY FILE ---
            src_file = os.path.join(DEAM_AUDIO_DIR, f"{song_id}.mp3")
            if os.path.exists(src_file):
                dst_file = os.path.join(OUTPUT_DIR, target_mood, f"deam_{song_id}.mp3")
                shutil.copy2(src_file, dst_file)
                count[target_mood] += 1
            else:
                skipped['missing_file'] += 1

        except Exception as e:
            continue

    print("\n" + "="*40)
    print("📊 LAPORAN DATA CLEANING DEAM")
    print("="*40)
    print(f"✅ Happy   : {count['happy']}")
    print(f"✅ Angry   : {count['angry']}")
    print(f"✅ Relaxed : {count['relaxed']}")
    print(f"✅ Sad     : {count['sad']}")
    print("-" * 40)
    print(f"🗑️ Dibuang (High STD/Perdebatan) : {skipped['high_std']} lagu")
    print(f"⚠️ Dibuang (Ambigu/Netral)     : {skipped['ambigu']} lagu")
    print(f"❌ Missing Files               : {skipped['missing_file']} lagu")
    print("="*40)

if __name__ == "__main__":
    prepare_deam_data()