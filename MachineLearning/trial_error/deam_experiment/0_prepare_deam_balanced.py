import os
import shutil
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# --- KONFIGURASI PATH ---
DEAM_AUDIO_DIR = r'./data/deam/audio' 
CSV_PATH_1 = r'./data/deam/annotations/static_annotations_averaged_songs_1_2000.csv'
CSV_PATH_2 = r'./data/deam/annotations/static_annotations_averaged_songs_2000_2058.csv'

OUTPUT_DIR = 'data/training'

def prepare_deam_balanced():
    print("⚖️ MENYIAPKAN DEAM DATASET (UNDERSAMPLING / BALANCED)...")
    
    # 1. Load & Merge CSV
    try:
        df1 = pd.read_csv(CSV_PATH_1)
        df2 = pd.read_csv(CSV_PATH_2)
        df = pd.concat([df1, df2], ignore_index=True)
        df.columns = df.columns.str.strip()
        print(f"✅ CSV Tergabung! Total Baris Awal: {len(df)}")
    except Exception as e:
        print(f"❌ Error Load CSV: {e}")
        return

    # 2. Tampung Data Dulu (Belum dicopy)
    # Kita butuh list semua lagu yang eligible dulu baru bisa dihitung minimumnya
    temp_storage = {
        'happy': [],
        'sad': [],
        'angry': [],
        'relaxed': []
    }
    
    # Parameter Filter (Sama seperti sebelumnya)
    BUFFER = 0.25
    MAX_STD = 2.5
    
    print("🔍 Scanning kandidat lagu...")
    
    for index, row in df.iterrows():
        try:
            song_id = int(row['song_id'])
            v_mean = float(row['valence_mean'])
            a_mean = float(row['arousal_mean'])
            v_std  = float(row['valence_std'])
            a_std  = float(row['arousal_std'])
            
            # Cek File Fisik Dulu (Kalau file mp3 gak ada, skip)
            src_file = os.path.join(DEAM_AUDIO_DIR, f"{song_id}.mp3")
            if not os.path.exists(src_file):
                continue

            # Filter STD (Konsistensi Manusia)
            if v_std > MAX_STD or a_std > MAX_STD:
                continue 

            # Filter Kuadran
            if v_mean > (5 + BUFFER) and a_mean > (5 + BUFFER):
                temp_storage['happy'].append(song_id)
            elif v_mean < (5 - BUFFER) and a_mean > (5 + BUFFER):
                temp_storage['angry'].append(song_id)
            elif v_mean < (5 - BUFFER) and a_mean < (5 - BUFFER):
                temp_storage['sad'].append(song_id)
            elif v_mean > (5 + BUFFER) and a_mean < (5 - BUFFER):
                temp_storage['relaxed'].append(song_id)

        except Exception:
            continue

    # 3. Cari Jumlah Terkecil (Min Count)
    counts = {k: len(v) for k, v in temp_storage.items()}
    min_count = min(counts.values())
    
    print("\n📊 Statistik Awal (Sebelum Pemangkasan):")
    for k, v in counts.items():
        print(f"   - {k.upper().ljust(8)}: {v}")
        
    print(f"\n✂️ LIMIT DITETAPKAN: {min_count} lagu per mood.")
    print(f"   (Semua mood akan dipotong acak menjadi {min_count} lagu)")

    # 4. Reset Folder Output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # 5. Eksekusi Copy (Hanya ambil sejumlah min_count)
    print("\n🚀 Menyalin File Terpilih...")
    
    final_count = 0
    
    for mood, song_ids in temp_storage.items():
        os.makedirs(os.path.join(OUTPUT_DIR, mood), exist_ok=True)
        
        # ACAK DULU BIAR FAIR (Shuffle)
        random.shuffle(song_ids)
        
        # AMBIL SEJUMLAH MIN_COUNT (Slicing)
        selected_songs = song_ids[:min_count]
        
        for song_id in tqdm(selected_songs, desc=f"   Copying {mood}"):
            src_file = os.path.join(DEAM_AUDIO_DIR, f"{song_id}.mp3")
            dst_file = os.path.join(OUTPUT_DIR, mood, f"deam_{song_id}.mp3")
            shutil.copy2(src_file, dst_file)
            final_count += 1

    print("\n" + "="*40)
    print("✅ DEAM BALANCED DATASET READY")
    print("="*40)
    print(f"🎯 Total Dataset : {final_count} file")
    print(f"🎯 Per Mood      : {min_count} file (SEIMBANG)")
    print("="*40)

if __name__ == "__main__":
    prepare_deam_balanced()