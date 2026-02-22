import os
import shutil
import random
from pathlib import Path

# --- KONFIGURASI ---
RAW_DIR = 'data/raw'
SPLIT_DIR = 'data/split'

# KITA SET TEST = 15, BERARTI TRAIN OTOMATIS JADI 10
TARGET_TEST_COUNT = 15 

def hybrid_split():
    print("⚖️ MENJALANKAN HYBRID SPLIT (10 Train / 15 Test)...")
    
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        src_path = os.path.join(RAW_DIR, mood)
        train_dst = os.path.join(SPLIT_DIR, 'train', mood)
        test_dst = os.path.join(SPLIT_DIR, 'test', mood)
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)
        
        files = [f for f in os.listdir(src_path) if f.endswith(('.mp3', '.wav'))]
        random.shuffle(files)
        
        # Ambil 15 untuk Test
        test_files = files[:TARGET_TEST_COUNT]
        # Sisanya (sekitar 10) untuk Train (Bocoran buat Model)
        train_files = files[TARGET_TEST_COUNT:]
        
        for f in test_files: shutil.copy2(os.path.join(src_path, f), os.path.join(test_dst, f))
        for f in train_files: shutil.copy2(os.path.join(src_path, f), os.path.join(train_dst, f))
            
        print(f"   📂 {mood.upper().ljust(8)} | Train (Masuk ke DEAM): {len(train_files)} | Test (Ujian): {len(test_files)}")

if __name__ == "__main__":
    hybrid_split()