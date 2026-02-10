import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

# --- KONFIGURASI ---
INPUT_TRAIN = 'data/training'  # MERGE Dataset (30s Clips)
INPUT_TEST  = 'data/raw'       # LOCAL Dataset (Full Songs)

OUTPUT_ROOT = 'data/interim_merge' 
OUTPUT_TRAIN = os.path.join(OUTPUT_ROOT, 'train')
OUTPUT_TEST  = os.path.join(OUTPUT_ROOT, 'test')

def trim_middle(y, sr, percentage=0.5):
    # Khusus untuk Lagu Full (Lokal)
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def add_noise(y): return y + 0.005 * np.random.randn(len(y))
def time_shift(y, sr): return np.roll(y, int(sr * 0.5))

def process(source, dest, is_train):
    if os.path.exists(dest): shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    for mood in moods:
        os.makedirs(os.path.join(dest, mood), exist_ok=True)
        src_dir = os.path.join(source, mood)
        if not os.path.exists(src_dir): continue
        
        files = [f for f in os.listdir(src_dir) if f.endswith(('.mp3', '.wav'))]
        
        # Tentukan mode: Train (MERGE) atau Test (Lokal)
        dataset_type = "MERGE (30s Full)" if is_train else "LOCAL (Middle 50%)"
        
        for f in tqdm(files, desc=f"Processing {mood} - {dataset_type}"):
            try:
                file_path = os.path.join(src_dir, f)
                y, sr = librosa.load(file_path, sr=22050)
                
                # --- LOGIKA KONDISIONAL ---
                if is_train:
                    # KASUS MERGE: Jangan potong! (Pakai full 30s)
                    # Kita cuma pastikan durasinya cukup, gak usah trim
                    pass 
                else:
                    # KASUS LOKAL: Lagu Full -> Potong 50% Tengah
                    y = trim_middle(y, sr, percentage=0.5)
                
                # Simpan
                name = os.path.splitext(f)[0]
                save_path = os.path.join(dest, mood, f"{name}.wav")
                sf.write(save_path, y, sr)
                
                # AUGMENTASI (Hanya untuk Training / MERGE)
                if is_train:
                    sf.write(os.path.join(dest, mood, f"{name}_n.wav"), add_noise(y), sr)
                    sf.write(os.path.join(dest, mood, f"{name}_s.wav"), time_shift(y, sr), sr)
                    
            except Exception as e:
                # print(f"Error {f}: {e}")
                pass

if __name__ == "__main__":
    print("🚀 PREPARING MERGE (FULL 30s) vs LOCAL (MIDDLE 50%)...")
    
    # 1. Proses MERGE (Ambil Full 30 detik + Augmentasi)
    process(INPUT_TRAIN, OUTPUT_TRAIN, is_train=True) 
    
    # 2. Proses LOKAL (Ambil Tengah Saja)
    process(INPUT_TEST, OUTPUT_TEST, is_train=False) 
    
    print("\n✅ SELESAI! Data siap di 'data/interim_merge'")
    print("   - Train: Full duration (karena sumbernya potongan)")
    print("   - Test : Middle 50% (karena sumbernya lagu full)")