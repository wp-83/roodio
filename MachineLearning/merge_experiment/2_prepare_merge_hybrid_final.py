import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

# --- KONFIGURASI ---
INPUT_TRAIN_MERGE = 'data/training'     # MERGE (30s)
INPUT_TRAIN_LOCAL = 'data/split/train'  # LOCAL TRAIN (10 Lagu)
INPUT_TEST_LOCAL  = 'data/split/test'   # LOCAL TEST (15 Lagu)

OUTPUT_ROOT = 'data/interim_hybrid_merge'
OUTPUT_TRAIN = os.path.join(OUTPUT_ROOT, 'train')
OUTPUT_TEST  = os.path.join(OUTPUT_ROOT, 'test')

# ⚠️ KRUSIAL: YAMNet Native Sample Rate
TARGET_SR = 16000 

def trim_middle(y, sr, percentage=0.5):
    """Mengambil bagian tengah audio (untuk data lokal full)."""
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def add_noise(y, noise_factor=0.005):
    """Menambahkan Gaussian Noise ringan."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift_random(y, sr, max_shift_sec=0.5):
    """Menggeser waktu secara acak (Random Rolling)."""
    # Shift acak antara -0.5 detik sampai +0.5 detik
    shift_samples = int(sr * max_shift_sec)
    shift = np.random.randint(-shift_samples, shift_samples)
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps):
    """Mengubah nada (Pitch Shifting)."""
    # Menggunakan library librosa yang robust
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def process_dataset(input_dirs, output_dir, is_train_mode, source_type="local"):
    if isinstance(input_dirs, str): input_dirs = [input_dirs]
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        out_mood_path = os.path.join(output_dir, mood)
        os.makedirs(out_mood_path, exist_ok=True)
        
        for input_dir in input_dirs:
            in_mood_path = os.path.join(input_dir, mood)
            if not os.path.exists(in_mood_path): continue
            
            prefix = "merge_" if source_type == "merge" else "local_"
            files = [f for f in os.listdir(in_mood_path) if f.endswith(('.mp3', '.wav'))]
            
            # Tampilkan progress bar yang jelas
            desc_text = f"{mood.upper()} - {source_type.upper()}"
            
            for f in tqdm(files, desc=desc_text, leave=False):
                try:
                    file_path = os.path.join(in_mood_path, f)
                    
                    # 1. LOAD dengan SR 16000 (Konsistensi YAMNet)
                    y, sr = librosa.load(file_path, sr=TARGET_SR)
                    
                    # 2. PREPROCESSING (Potong atau Tidak)
                    if source_type == "merge":
                        # MERGE: Biarkan Full (karena sudah potongan 30s)
                        pass 
                    else:
                        # LOCAL: Potong 50% Tengah (buang intro/outro)
                        y = trim_middle(y, sr, percentage=0.5)
                    
                    # Pastikan audio tidak kosong setelah dipotong
                    if len(y) < 1000: continue

                    base_name = os.path.splitext(f)[0]
                    save_name = f"{prefix}{base_name}"
                    
                    # 3. SIMPAN ORIGINAL (Wajib ada)
                    sf.write(os.path.join(out_mood_path, f"{save_name}.wav"), y, sr)
                    
                    # 4. AUGMENTASI (Hanya Training)
                    if is_train_mode:
                        # A. Noise (Semua data train dapat noise)
                        y_noise = add_noise(y)
                        sf.write(os.path.join(out_mood_path, f"{save_name}_n.wav"), y_noise, sr)
                        
                        # B. Time Shift Random (Semua data train dapat shift)
                        y_shift = time_shift_random(y, sr)
                        sf.write(os.path.join(out_mood_path, f"{save_name}_s.wav"), y_shift, sr)
                        
                        # C. PITCH SHIFT (KHUSUS DATA LOKAL)
                        # Agar 10 lagu lokal terasa lebih variatif
                        if source_type == "local":
                            # Pitch Up +1 Semitone (Aman)
                            y_up = pitch_shift(y, sr, n_steps=1.0)
                            sf.write(os.path.join(out_mood_path, f"{save_name}_p1.wav"), y_up, sr)
                            
                            # Pitch Down -1 Semitone (Aman)
                            y_down = pitch_shift(y, sr, n_steps=-1.0)
                            sf.write(os.path.join(out_mood_path, f"{save_name}_m1.wav"), y_down, sr)
                            
                except Exception as e:
                    # print(f"Error processing {f}: {e}")
                    pass

def main():
    print("🛡️ PREPARING RESEARCH-GRADE HYBRID DATASET")
    print("   Specs: SR=16kHz, Pitch=±1, Random Shift")
    print("="*60)
    
    # 1. TRAIN: MERGE (Full 30s + Basic Augment)
    process_dataset(INPUT_TRAIN_MERGE, OUTPUT_TRAIN, is_train_mode=True, source_type="merge")
    
    # 2. TRAIN: LOCAL (Middle 50% + Advanced Augment)
    process_dataset(INPUT_TRAIN_LOCAL, OUTPUT_TRAIN, is_train_mode=True, source_type="local")
    
    # 3. TEST: LOCAL (Middle 50% Murni - NO Augment)
    process_dataset(INPUT_TEST_LOCAL, OUTPUT_TEST, is_train_mode=False, source_type="local")
    
    print("\n✅ DATA READY: data/interim_hybrid_merge")
    print("   Siap untuk Training YAMNet (Pastikan SR di script training juga 16000)")

if __name__ == "__main__":
    main()