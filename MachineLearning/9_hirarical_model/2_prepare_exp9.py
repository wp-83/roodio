import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
import random # Tambahkan ini

# --- 1. KUNCI RANDOM (SEED) ---
# Ini menjamin pola noise dan geseran waktu SELALU SAMA setiap kali di-run
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(43) # Panggil di paling atas!

# --- KONFIGURASI ---
INPUT_DIR = 'data/split_exp9'
OUTPUT_DIR = 'data/processed_exp9' 
TARGET_SR = 16000 

def trim_middle(y, sr, percentage=0.5):
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# Karena np.random sudah di-seed, pola noise ini akan konsisten
def add_noise(y): return y + 0.005 * np.random.randn(len(y))

# Karena np.random sudah di-seed, nilai shift ini akan konsisten
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
            
            # PENTING: Gunakan sorted()!
            # os.listdir() urutannya bisa berubah-ubah di tiap komputer.
            # sorted() memastikan file diproses dengan urutan abjad yang pasti,
            # sehingga urutan pemakaian random seed juga pasti.
            files = sorted(os.listdir(src_path))
            
            for f in tqdm(files, desc=f"{subset.upper()} - {mood}", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR)
                    y = trim_middle(y, sr, 0.5) 
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # 1. Simpan Original
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                    
                    # 2. Augmentasi (HANYA TRAIN)
                    if subset == 'train':
                        sf.write(os.path.join(dst_path, f"{base_name}_n.wav"), add_noise(y), sr)
                        sf.write(os.path.join(dst_path, f"{base_name}_s.wav"), time_shift(y, sr), sr)
                        
                except Exception: pass

if __name__ == "__main__":
    print("🚀 PROCESSING EXP 9 (FIXED SEED)...")
    process_dataset()
    print(f"✅ Data Siap di: {OUTPUT_DIR}")