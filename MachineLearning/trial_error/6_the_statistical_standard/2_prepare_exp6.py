import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
import random

# --- 1. KUNCI RANDOM (SEED) ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(42)

# --- KONFIGURASI ---
# Kita ambil dari split yang SUDAH DICLEANING (Exp 4)
INPUT_DIR = 'data/split_exp4' 
OUTPUT_DIR = 'data/processed_exp6' # Folder Baru Khusus Exp 6
TARGET_SR = 16000 

def trim_middle(y, sr, percentage=0.5):
    """Mengambil n% bagian tengah audio"""
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def normalize_loudness(y):
    """Peak Normalization: Memastikan volume max menyentuh -1.0 s/d 1.0"""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def process_dataset():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    # Kita proses Train dan Test
    for subset in ['train', 'test']:
        moods = ['happy', 'sad', 'angry', 'relaxed']
        for mood in moods:
            src_path = os.path.join(INPUT_DIR, subset, mood)
            dst_path = os.path.join(OUTPUT_DIR, subset, mood)
            os.makedirs(dst_path, exist_ok=True)
            
            if not os.path.exists(src_path): continue
            
            # Sorted agar urutan deterministik
            files = sorted(os.listdir(src_path))
            
            for f in tqdm(files, desc=f"{subset.upper()} - {mood}", leave=False):
                if not f.endswith(('.wav', '.mp3')): continue
                
                try:
                    # 1. Resample ke 16kHz & Mono (Otomatis oleh librosa)
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR, mono=True)
                    
                    # 2. Trim Middle (Ambil 50% tengah)
                    y = trim_middle(y, sr, 0.5) 
                    
                    # 3. Normalize Loudness
                    y = normalize_loudness(y)
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # Simpan File Bersih
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                    
                    # Catatan: Di Exp 6 ini kita TIDAK melakukan Augmentasi (Noise/Shift)
                    # Kita ingin menguji kemurni data + fitur MeanStd.
                        
                except Exception as e:
                    print(f"⚠️ Skip {f}: {e}")

if __name__ == "__main__":
    print("🚀 PREPARING EXPERIMENT 6 (Minimal Ideal)...")
    process_dataset()
    print(f"✅ Data Siap di: {OUTPUT_DIR}")