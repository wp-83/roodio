import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

# --- KONFIGURASI PATH ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1] # Sesuaikan jika lokasi script beda

# 1. SUMBER DATA
# Train ambil dari DEAM (Folder yang baru saja Anda buat)
INPUT_TRAIN_DEAM = 'data/training' 
# Test ambil dari Data Lokal Anda (Folder raw lama Anda)
INPUT_TEST_LOCAL = 'data/raw'

# 2. TUJUAN DATA (Folder siap makan untuk YAMNet)
OUTPUT_ROOT = 'data/interim_correct'
OUTPUT_TRAIN = os.path.join(OUTPUT_ROOT, 'train')
OUTPUT_TEST  = os.path.join(OUTPUT_ROOT, 'test')

def trim_middle(y, sr, percentage=0.5):
    """Ambil 50% bagian tengah lagu (buang intro/outro)"""
    total_samples = len(y)
    keep_samples = int(total_samples * percentage)
    start_idx = int((total_samples - keep_samples) / 2)
    end_idx = start_idx + keep_samples
    
    y_trimmed = y[start_idx:end_idx]
    if len(y_trimmed) < sr: return y # Safety check
    return y_trimmed

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, sr, shift_max=2.0):
    shift = np.random.randint(sr * shift_max)
    return np.roll(y, shift)

def process_dataset(input_dir, output_dir, is_train=True):
    mode_name = "TRAINING (DEAM + Augment)" if is_train else "TESTING (Lokal + Original)"
    print(f"\n🚀 Memproses: {mode_name}")
    print(f"   📂 Source: {input_dir}")
    print(f"   📂 Dest  : {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Folder input tidak ditemukan: {input_dir}")
        return

    # Reset folder output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        in_mood_path = os.path.join(input_dir, mood)
        out_mood_path = os.path.join(output_dir, mood)
        
        if not os.path.isdir(in_mood_path): continue
        os.makedirs(out_mood_path, exist_ok=True)
        
        files = [f for f in os.listdir(in_mood_path) if f.endswith(('.mp3', '.wav', '.flac'))]
        print(f"   Processing {mood}: {len(files)} files...")
        
        for f in tqdm(files):
            try:
                # 1. Load Audio
                file_path = os.path.join(in_mood_path, f)
                # Resample ke 22050 dulu standar librosa, nanti YAMNet urus sisanya
                y, sr = librosa.load(file_path, sr=22050)
                
                # 2. POTONG TENGAH (Middle 50%)
                # Ini kunci sukses kita berdasarkan EDA kemarin
                y = trim_middle(y, sr, percentage=0.5)
                
                base_name = os.path.splitext(f)[0]
                
                # 3. Simpan Versi Original (Trimmed)
                # Baik Train maupun Test wajib punya versi bersih ini
                sf.write(os.path.join(out_mood_path, f"{base_name}.wav"), y, sr)
                
                # 4. Augmentasi (HANYA UNTUK TRAIN / DEAM)
                if is_train:
                    # Noise
                    y_noise = add_noise(y)
                    sf.write(os.path.join(out_mood_path, f"{base_name}_noise.wav"), y_noise, sr)
                    
                    # Shift
                    y_shift = time_shift(y, sr)
                    sf.write(os.path.join(out_mood_path, f"{base_name}_shift.wav"), y_shift, sr)
                    
            except Exception as e:
                # Error sesekali wajar di dataset besar
                pass

def main():
    print("🛡️ PREPARING CROSS-CORPUS DATASET")
    print("="*40)
    
    # 1. Proses TRAIN (DEAM)
    # Input: data/training (DEAM) -> Output: data/interim_correct/train
    process_dataset(INPUT_TRAIN_DEAM, OUTPUT_TRAIN, is_train=True)
    
    # 2. Proses TEST (Lokal)
    # Input: data/split/test (Lokal) -> Output: data/interim_correct/test
    process_dataset(INPUT_TEST_LOCAL, OUTPUT_TEST, is_train=False)
    
    print("\n✅ SELESAI!")
    print(f"   Data siap di: {OUTPUT_ROOT}")
    print("   Silakan jalankan '4_train_yamnet.py' sekarang.")

if __name__ == "__main__":
    main()