import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

# Input: Data Augmentasi (Train) & Data Original (Test)
INPUT_ROOT = PROJECT_DIR / 'data' / 'interim_correct'

# Output: Folder Segmen Siap Train
OUTPUT_ROOT = PROJECT_DIR / 'data' / 'processed_yamnet_segments'

SEGMENT_SECONDS = 3  # Durasi per segmen
TARGET_SR = 16000    # YAMNet WAJIB 16kHz

def process_slicing():
    print("🔪 SLICING FOR YAMNET (3 Detik @ 16kHz)...")
    
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT) # Reset biar bersih
    
    for subset in ['train', 'test']:
        input_subset = INPUT_ROOT / subset
        output_subset = OUTPUT_ROOT / subset
        
        if not input_subset.exists(): continue
        
        moods = os.listdir(input_subset)
        
        for mood in moods:
            in_mood_path = input_subset / mood
            out_mood_path = output_subset / mood
            
            if not in_mood_path.is_dir(): continue
            os.makedirs(out_mood_path, exist_ok=True)
            
            files = [f for f in os.listdir(in_mood_path) if f.endswith('.wav')]
            
            print(f"   📂 Processing {subset}/{mood} ({len(files)} songs)...")
            
            for f in tqdm(files):
                try:
                    # 1. Load & Resample langsung ke 16k
                    file_path = in_mood_path / f
                    y, sr = librosa.load(file_path, sr=TARGET_SR)
                    
                    # 2. Potong-potong
                    samples_per_seg = SEGMENT_SECONDS * TARGET_SR
                    total_segments = len(y) // samples_per_seg
                    
                    # Skip jika lagu terlalu pendek (< 3 detik)
                    if total_segments == 0:
                        # Simpan apa adanya jika kependekan
                        sf.write(out_mood_path / f, y, TARGET_SR)
                        continue
                        
                    base_name = os.path.splitext(f)[0]
                    
                    for i in range(total_segments):
                        start = i * samples_per_seg
                        end = start + samples_per_seg
                        segment = y[start:end]
                        
                        # Simpan segmen
                        # Nama file: laguAsli_seg0.wav, laguAsli_seg1.wav
                        # Ini penting agar kita tahu segmen ini milik siapa
                        out_name = f"{base_name}_seg{i}.wav"
                        sf.write(out_mood_path / out_name, segment, TARGET_SR)
                        
                except Exception as e:
                    print(f"❌ Error {f}: {e}")

    print("\n✅ SELESAI! Data segments tersimpan di:", OUTPUT_ROOT)

if __name__ == "__main__":
    process_slicing()