import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

# --- KONFIGURASI PATH ---
CURRENT_FILE = Path(__file__).resolve()

# 1. SUMBER DATA TRAIN (GABUNGAN)
INPUT_TRAIN_SOURCES = [
    'data/training',      # DEAM (Ribuan lagu)
    'data/split/train'    # LOKAL (10 lagu "bocoran")
]

# 2. SUMBER DATA TEST (MURNI)
INPUT_TEST_SOURCE = 'data/split/test' # LOKAL (15 lagu ujian)

# 3. OUTPUT
OUTPUT_ROOT = 'data/interim_correct'
OUTPUT_TRAIN = os.path.join(OUTPUT_ROOT, 'train')
OUTPUT_TEST  = os.path.join(OUTPUT_ROOT, 'test')

def trim_middle(y, sr, percentage=0.5):
    total_samples = len(y)
    keep_samples = int(total_samples * percentage)
    start_idx = int((total_samples - keep_samples) / 2)
    end_idx = start_idx + keep_samples
    return y[start_idx:end_idx]

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, sr, shift_max=2.0):
    shift = np.random.randint(sr * shift_max)
    return np.roll(y, shift)

def process_dataset(input_sources, output_dir, is_train=True):
    if isinstance(input_sources, str): input_sources = [input_sources]
    
    print(f"\n🚀 Memproses ke: {output_dir}")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        out_mood_path = os.path.join(output_dir, mood)
        os.makedirs(out_mood_path, exist_ok=True)
        
        # Loop semua sumber (misal: DEAM + Lokal)
        for input_dir in input_sources:
            in_mood_path = os.path.join(input_dir, mood)
            if not os.path.exists(in_mood_path): continue
            
            # Label prefix biar tau asalnya darimana
            prefix = "deam_" if "training" in input_dir else "local_"
            
            files = [f for f in os.listdir(in_mood_path) if f.endswith(('.mp3', '.wav'))]
            
            for f in tqdm(files, desc=f"   {mood} from {os.path.basename(input_dir)}", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(in_mood_path, f), sr=22050)
                    y = trim_middle(y, sr, percentage=0.5) # Middle 50% Strategy
                    
                    base_name = os.path.splitext(f)[0]
                    save_name = f"{prefix}{base_name}"
                    
                    # Simpan Original
                    sf.write(os.path.join(out_mood_path, f"{save_name}.wav"), y, sr)
                    
                    # Augmentasi (HANYA UNTUK TRAIN)
                    if is_train:
                        sf.write(os.path.join(out_mood_path, f"{save_name}_noise.wav"), add_noise(y), sr)
                        sf.write(os.path.join(out_mood_path, f"{save_name}_shift.wav"), time_shift(y, sr), sr)
                        
                except Exception:
                    pass

def main():
    print("🛡️ PREPARING HYBRID DATASET (DEAM + LOCAL MIX)")
    print("="*50)
    
    # 1. TRAIN: DEAM + 10 Lokal (Augmented)
    process_dataset(INPUT_TRAIN_SOURCES, OUTPUT_TRAIN, is_train=True)
    
    # 2. TEST: 15 Lokal (Original)
    process_dataset(INPUT_TEST_SOURCE, OUTPUT_TEST, is_train=False)
    
    print("\n✅ SELESAI! Siap Training.")

if __name__ == "__main__":
    main()