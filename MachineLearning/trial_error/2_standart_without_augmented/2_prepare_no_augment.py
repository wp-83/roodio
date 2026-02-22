import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil

# --- KONFIGURASI ---
# Kita gunakan sumber split yang SAMA dengan eksperimen sebelumnya biar adil
INPUT_DIR = 'data/split_basic' 
OUTPUT_DIR = 'data/processed_no_augment' # Folder output baru
TARGET_SR = 16000 

def trim_middle(y, sr, percentage=0.5):
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def process_dataset():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    # Loop Train & Test
    for subset in ['train', 'test']:
        moods = ['happy', 'sad', 'angry', 'relaxed']
        for mood in moods:
            src_path = os.path.join(INPUT_DIR, subset, mood)
            dst_path = os.path.join(OUTPUT_DIR, subset, mood)
            os.makedirs(dst_path, exist_ok=True)
            
            if not os.path.exists(src_path): continue
            
            files = os.listdir(src_path)
            for f in tqdm(files, desc=f"{subset.upper()} - {mood} (No Augment)", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR)
                    
                    # HANYA POTONG TENGAH (Satu-satunya preprocessing)
                    y = trim_middle(y, sr, 0.5) 
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # Simpan File Asli Saja
                    # TIDAK ADA AUGMENTASI SAMA SEKALI DISINI
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                        
                except Exception: pass

if __name__ == "__main__":
    print("🚀 PREPARING DATA (PURE RAW - NO AUGMENTATION)...")
    process_dataset()
    print(f"✅ Data Siap di: {OUTPUT_DIR}")