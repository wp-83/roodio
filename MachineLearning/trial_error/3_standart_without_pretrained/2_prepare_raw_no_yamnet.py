import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil

# --- KONFIGURASI ---
INPUT_DIR = 'data/split_basic'       # Gunakan split yang sama biar adil
OUTPUT_DIR = 'data/processed_manual' # Folder output baru
TARGET_SR = 22050 # MFCC standar biasanya 22050Hz (bukan 16000Hz kyk YAMNet)

def trim_middle(y, sr, percentage=0.5):
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

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
            for f in tqdm(files, desc=f"{subset.upper()} - {mood}", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR)
                    
                    # HANYA POTONG TENGAH (MURNI)
                    y = trim_middle(y, sr, 0.5) 
                    
                    base_name = os.path.splitext(f)[0]
                    # Simpan
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                        
                except Exception: pass

if __name__ == "__main__":
    print("🚀 PREPARING RAW DATA (FOR MANUAL MFCC)...")
    process_dataset()
    print(f"✅ Data Siap di: {OUTPUT_DIR}")