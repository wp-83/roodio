import os
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

# Input tetap dari Lagu Full (Raw)
INPUT_ROOT = PROJECT_DIR / 'data' / 'interim_correct' 

# Output ke folder BARU (Biar bisa dibandingin sama yang lama)
OUTPUT_ROOT = PROJECT_DIR / 'data' / 'processed_cnn_middle'

SEGMENT_LEN = 3 # Detik
SR = 22050      

def process_data():
    print("🔪 FOKUS TENGAH: Memotong 50% tengah lagu & Konversi ke Spektrogram...")
    print(f"   Sumber: {INPUT_ROOT}")
    print(f"   Tujuan: {OUTPUT_ROOT}")
    
    if not INPUT_ROOT.exists():
        print("❌ Error: Folder Input tidak ditemukan!")
        return

    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Loop Train & Test
    for subset in ['train', 'test']:
        input_dir = INPUT_ROOT / subset
        output_dir = OUTPUT_ROOT / subset
        
        if not input_dir.exists(): continue
        
        for mood in os.listdir(input_dir):
            mood_in = input_dir / mood
            mood_out = output_dir / mood
            
            if not os.path.isdir(mood_in): continue
            
            os.makedirs(mood_out, exist_ok=True)
            
            print(f"   📂 Processing {subset}/{mood} (Middle Only)...")
            
            files = [f for f in os.listdir(mood_in) if f.endswith(('.mp3', '.wav', '.flac'))]
            
            for f in tqdm(files):
                file_path = mood_in / f
                try:
                    # 1. Load Audio Full
                    y, _ = librosa.load(file_path, sr=SR)
                    
                    if len(y) == 0: continue

                    # --- LOGIKA "AMBIL TENGAH" ---
                    total_samples = len(y)
                    start_idx = int(total_samples * 0.25) # Buang 25% awal
                    end_idx = int(total_samples * 0.75)   # Buang 25% akhir
                    
                    # Ambil dagingnya saja
                    y_middle = y[start_idx:end_idx]
                    
                    # Cek apakah sisa potongannya cukup (minimal 3 detik)
                    samples_per_segment = SEGMENT_LEN * SR
                    if len(y_middle) < samples_per_segment:
                        # Kalau lagu kependekan, pakai semua aja
                        y_middle = y 
                    
                    # 2. Hitung segmen dari bagian tengah
                    total_segments = len(y_middle) // samples_per_segment
                    
                    # 3. Potong & Buat Spektrogram
                    for i in range(total_segments):
                        start = i * samples_per_segment
                        end = start + samples_per_segment
                        segment = y_middle[start:end]
                        
                        # Mel-Spectrogram
                        melspec = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=128)
                        melspec_db = librosa.power_to_db(melspec, ref=np.max)
                        
                        # Normalisasi 0-1
                        min_val = melspec_db.min()
                        max_val = melspec_db.max()
                        if max_val - min_val == 0:
                            melspec_norm = np.zeros_like(melspec_db)
                        else:
                            melspec_norm = (melspec_db - min_val) / (max_val - min_val)
                        
                        # Simpan
                        out_name = f"{os.path.splitext(f)[0]}_mid_seg{i}.npy"
                        np.save(mood_out / out_name, melspec_norm)
                        
                except Exception as e:
                    print(f"❌ Error {f}: {e}")

if __name__ == "__main__":
    process_data()