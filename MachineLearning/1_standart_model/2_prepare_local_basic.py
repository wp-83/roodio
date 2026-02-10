import os
import librosa
import soundfile as sf
import numpy as np
import random # Tambahkan ini
from tqdm import tqdm
import shutil

# --- BAGIAN PENGUNCI RANDOM (SEED) ---
def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(42) # Panggil di paling atas!

# --- KONFIGURASI ---
INPUT_DIR = 'data/split_basic'
OUTPUT_DIR = 'data/processed_basic' # Output ke folder terpisah
TARGET_SR = 16000 # Standard YAMNet

def trim_middle(y, sr, percentage=0.5):
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# Karena np.random sudah di-seed, hasil noise ini akan konsisten
def add_noise(y): return y + 0.005 * np.random.randn(len(y))

# Karena np.random sudah di-seed, geseran waktu akan konsisten
def time_shift(y, sr): 
    shift = np.random.randint(-int(sr*0.5), int(sr*0.5))
    return np.roll(y, shift)

def process_dataset():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    for subset in ['train', 'test']:
        moods = ['happy', 'sad', 'angry', 'relaxed']
        for mood in moods:
            src_path = os.path.join(INPUT_DIR, subset, mood)
            dst_path = os.path.join(OUTPUT_DIR, subset, mood)
            os.makedirs(dst_path, exist_ok=True)
            
            if not os.path.exists(src_path): continue
            
            files = os.listdir(src_path)
            # Urutkan file agar urutan processing selalu sama
            files.sort() 
            
            for f in tqdm(files, desc=f"{subset.upper()} - {mood}", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR)
                    y = trim_middle(y, sr, 0.5) # Middle 50%
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # 1. Simpan Original
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                    
                    # 2. Augmentasi (HANYA TRAIN)
                    if subset == 'train':
                        sf.write(os.path.join(dst_path, f"{base_name}_n.wav"), add_noise(y), sr)
                        sf.write(os.path.join(dst_path, f"{base_name}_s.wav"), time_shift(y, sr), sr)
                        
                except Exception: pass

if __name__ == "__main__":
    print("🚀 PROCESSING DATA BASIC (FIXED SEED)...")
    process_dataset()
    print(f"✅ Data Siap di: {OUTPUT_DIR}")