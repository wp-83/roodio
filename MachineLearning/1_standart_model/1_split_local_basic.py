import os
import shutil
import random
import numpy as np # Tambahkan ini
from pathlib import Path

# --- BAGIAN PENGUNCI RANDOM (SEED) ---
def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(42) # Panggil di paling atas!

# --- KONFIGURASI ---
RAW_DIR = 'data/raw'
SPLIT_DIR = 'data/split_basic'

# Total 25 lagu/mood. Kita ambil 7 untuk Test, sisanya (18) untuk Train.
TARGET_TEST_PER_MOOD = 7 

def split_local_basic():
    print("⚖️ SPLITTING DATA LOCAL (72 Train / 28 Test)...")
    
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        src_path = os.path.join(RAW_DIR, mood)
        train_dst = os.path.join(SPLIT_DIR, 'train', mood)
        test_dst = os.path.join(SPLIT_DIR, 'test', mood)
        
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)
        
        # Ambil semua file audio
        files = [f for f in os.listdir(src_path) if f.endswith(('.mp3', '.wav'))]
        
        # KARENA SEED DIKUNCI, HASIL SHUFFLE INI AKAN SELALU SAMA
        random.shuffle(files) 
        
        # Split
        test_files = files[:TARGET_TEST_PER_MOOD]
        train_files = files[TARGET_TEST_PER_MOOD:]
        
        # Copy
        for f in test_files: shutil.copy2(os.path.join(src_path, f), os.path.join(test_dst, f))
        for f in train_files: shutil.copy2(os.path.join(src_path, f), os.path.join(train_dst, f))
            
        print(f"   📂 {mood.upper().ljust(8)} | Train: {len(train_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    split_local_basic()